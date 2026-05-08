import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from enum import Enum
from typing import List, Dict, Tuple, Any, Optional
from cpgqls_client import CPGQLSClient, import_code_query, delete_query


class QueryStatus(Enum):
    SUCCESSFUL = 1
    EMPTYRESULT = 2
    ERROR = 3


class JoernManager:
    """
    Manages all interactions with Joern server, including:
    - Server lifecycle (start, stop, restart as local processes)
    - Running queries
    - Loading and deleting projects
    - Extracting paths and other data from Joern

    Replaces Docker-based Joern management with direct subprocess management.
    """

    def __init__(self, port: int, joern_path: str):
        """
        Initialize a Joern Manager for a specific port

        Args:
            port: Joern server port number
            joern_path: Path to the joern-cli directory containing the 'joern' binary
        """
        self.port = port
        self.joern_path = joern_path
        self.server_name = f"joern_server_{port}"
        self.joern_client = CPGQLSClient(f"localhost:{port}")
        self._process: Optional[subprocess.Popen] = None
        self._stderr_file: Optional[Any] = None
        self._project_dirs: Dict[str, str] = {}

    def _get_joern_bin(self) -> str:
        """Get the path to the joern binary within the joern-cli directory."""
        return os.path.join(self.joern_path, "joern")

    def start_server(self) -> bool:
        """
        Start a Joern server as a local subprocess on the configured port.

        Returns:
            Boolean indicating if the server started successfully and is healthy
        """
        joern_bin = self._get_joern_bin()

        if not os.path.isfile(joern_bin):
            print(f"Joern binary not found at {joern_bin}")
            return False

        cmd = [
            joern_bin, "--server",
            "--server-host", "0.0.0.0",
            "--server-port", str(self.port)
        ]

        try:
            # Capture stderr to a temp file so we can diagnose startup errors
            self._stderr_file = tempfile.NamedTemporaryFile(
                mode="w+", prefix=f"joern_{self.port}_", suffix=".log", delete=False
            )
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
                cwd=self.joern_path
            )

            # Give Joern's JVM time to initialize before the first health check.
            # Without this delay, the first check almost always hits a
            # ConnectionRefusedError because the server isn't listening yet.
            print(f"Joern process started (PID {self._process.pid}), "
                  f"waiting for JVM initialization...")
            for waited in range(10):
                time.sleep(1)
                if self._process.poll() is not None:
                    self._print_stderr_tail()
                    print(f"Joern process exited prematurely with code "
                          f"{self._process.returncode}")
                    return False
                # Probe early: if the port is already listening, start
                # health checks immediately instead of waiting the full 10 s.
                if waited >= 4 and self.check_server_health():
                    print(f"Joern server {self.server_name} is ready early "
                          f"(after ~{waited + 1}s).")
                    return True

            return self._wait_for_server_health()
        except Exception as e:
            print(f"Error starting Joern server on port {self.port}: {e}")
            self._print_stderr_tail()
            return False

    def stop_server(self):
        """Stop the Joern server subprocess gracefully, then forcefully if needed."""
        if self._process is None:
            return

        try:
            self._process.terminate()
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        except Exception:
            pass
        finally:
            self._process = None
            if self._stderr_file is not None:
                try:
                    self._stderr_file.close()
                    os.unlink(self._stderr_file.name)
                except Exception:
                    pass
                self._stderr_file = None

    def _print_stderr_tail(self, lines: int = 30):
        """Print the last N lines of the Joern stderr log for debugging."""
        if self._stderr_file is None:
            return
        try:
            self._stderr_file.flush()
            with open(self._stderr_file.name, "r") as f:
                content = f.read()
            if content.strip():
                tail = "\n".join(content.splitlines()[-lines:])
                print(f"Joern stderr (last {lines} lines):\n{tail}")
        except Exception:
            pass

    def check_server_health(self) -> bool:
        """
        Check if the Joern server is healthy by running a simple query

        Returns:
            Boolean indicating if server is healthy
        """
        try:
            status, _ = self.run_query("val x = 1")
            return status == QueryStatus.SUCCESSFUL
        except Exception as e:
            if not getattr(self, "_health_failure_logged", False):
                self._health_failure_logged = True
                print(f"Joern health check failed ({type(e).__name__}: {e})")
            return False

    def recreate_server(self) -> bool:
        """
        Recreate the Joern server to address memory issues.
        Kills the existing process and starts a fresh one.

        Returns:
            Boolean indicating if server recreation was successful
        """
        try:
            # Initial small delay to prevent overwhelming the system
            time.sleep(2)
            print(f"Starting recreation of server: {self.server_name}")

            self.stop_server()
            is_healthy = self.start_server()

            if is_healthy:
                print(f"Server {self.server_name} recreated and verified successfully")
            else:
                print(f"Server {self.server_name} may not be fully operational")

            return is_healthy
        except Exception as e:
            print(f"Unexpected error with service {self.server_name}: {e}")
            return False

    def _wait_for_server_health(self, max_wait: int = 120, check_interval: int = 5) -> bool:
        """
        Wait for a service to be fully operational

        Args:
            max_wait: Maximum time to wait (in seconds)
            check_interval: Time between health checks (in seconds)

        Returns:
            Boolean indicating if service became healthy
        """
        total_waited = 0
        self._health_failure_logged = False

        while total_waited < max_wait:
            # If Joern process died, don't keep waiting
            if self._process is not None and self._process.poll() is not None:
                print(f"Joern process exited with code {self._process.returncode}")
                self._print_stderr_tail()
                return False

            try:
                if self.check_server_health():
                    print(f"Joern server {self.server_name} is ready to accept requests.")
                    self._health_failure_logged = False
                    return True
            except Exception as e:
                print(f"Error checking health for {self.server_name}: {e}")

            time.sleep(check_interval)
            total_waited += check_interval

        print(f"Server {self.server_name} did not become healthy within {max_wait} seconds")
        self._print_stderr_tail()
        return False

    def run_query(self, query: str) -> Tuple[QueryStatus, str]:
        """
        Run a Joern query on the CPG of source code

        Args:
            query: The CPGQL query to execute

        Returns:
            Tuple containing query status and output
        """
        print("Query --->", query)
        result = self.joern_client.execute(query)
        print("result: ", result)
        stdout = result["stdout"]

        if "Error" in stdout or "ConsoleException" in stdout:
            return QueryStatus.ERROR, stdout
        elif "List()" in stdout or "= empty iterator" in stdout:
            return QueryStatus.EMPTYRESULT, stdout
        else:
            return QueryStatus.SUCCESSFUL, stdout

    def run_queries(self, queries: list, source_code: str) -> Tuple[bool, list]:
        """
        Run a sequence of queries and extract paths from the last query result

        Args:
            queries: List of CPGQL queries to execute
            source_code: The source code being analyzed

        Returns:
            Tuple containing success flag and extracted paths
        """
        print("Running queries...")
        # Running each query except the last one
        for query in queries[:-1]:
            try:
                status, _ = self.run_query(query)
                if status == QueryStatus.ERROR:
                    return False, []
            except Exception as e:
                print(f"Failed to run query: {e}, moving to the next sample")
                return False, []

        # Extract paths from the last query
        success, paths = self.extract_joern_paths(source_code, queries)
        return success, paths

    def load_project(self, folder_path: str, code_content: Optional[str] = None) -> str:
        """
        Load a project into Joern.

        Writes *code_content* to a temporary directory so that Joern has a real
        filesystem path to import, then cleans up that directory on delete_project.

        Args:
            folder_path: The filename to use for the Joern project (e.g. "foo.c").
            code_content: The source code to write. Required — if omitted the old
                          behaviour is preserved (the path itself is assumed to exist).

        Returns:
            Output from the import operation
        """
        if code_content is not None:
            project_dir = tempfile.mkdtemp(prefix="joern_proj_")
            code_file = os.path.join(project_dir, folder_path)
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code_content)
            self._project_dirs[folder_path] = project_dir
            import_path = os.path.abspath(project_dir)
        else:
            import_path = folder_path

        import_code_qr = import_code_query(import_path, folder_path)
        print("import query: ", import_code_qr)
        status, stdout = self.run_query(import_code_qr)
        print("stdout: ", stdout)
        print(f"Project loaded from {import_path}")
        return stdout

    def delete_project(self, project_name: str) -> str:
        """
        Delete a project from Joern and clean up any leftover directories.

        Args:
            project_name: Name of the project to delete

        Returns:
            Output from the delete operation
        """
        delete_project_query = delete_query(project_name)
        print("delete query: ", delete_project_query)
        status, stdout = self.run_query(delete_project_query)
        print("stdout: ", stdout)

        # 1. Remove the temp directory we created in load_project.
        project_dir = self._project_dirs.pop(project_name, None)
        if project_dir and os.path.isdir(project_dir):
            try:
                shutil.rmtree(project_dir, ignore_errors=True)
                print(f"Removed temp project dir: {project_dir}")
            except Exception as exc:
                print(f"Failed to remove temp project dir {project_dir}: {exc}")

        # 2. Joern may leave behind workspace/<project_name> directories
        #    when its internal delete fails. Parse the error message and
        #    remove them ourselves.
        if self._looks_like_delete_failure(stdout):
            self._cleanup_joern_workspace_dirs(project_name, stdout)

        print(f"Project {project_name} deleted")
        return stdout

    @staticmethod
    def _looks_like_delete_failure(stdout: str) -> bool:
        """Return True if *stdout* indicates Joern could not fully delete the project."""
        if not stdout:
            return False
        markers = [
            "Cannot delete", "Could not delete", "Failed to delete",
            "Unable to remove", "Unable to delete",
            "ConsoleException",
        ]
        return any(m in stdout for m in markers)

    def _cleanup_joern_workspace_dirs(self, project_name: str, stdout: str):
        """Remove workspace directories Joern failed to delete, guided by the error message."""
        # Joern typically creates projects under <joern_cwd>/workspace/<name>
        candidates = [
            os.path.join(self.joern_path, "workspace", project_name),
        ]

        # Also try to pull any explicit paths from Joern's stderr/stdout
        path_pattern = re.compile(r"['\"]?((?:/[^\s'\"]+)|(?:[A-Za-z]:\\[^\s'\"]+))['\"]?")
        for match in path_pattern.finditer(stdout):
            p = match.group(1)
            if os.path.exists(p):
                candidates.append(p)

        for path in candidates:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"Manually removed Joern workspace dir: {path}")
                except Exception as exc:
                    print(f"Failed to remove Joern workspace dir {path}: {exc}")

    def extract_joern_paths(self, source_code: str, queries: list) -> Tuple[bool, list]:
        """
        Extract paths from Joern analysis results

        Args:
            source_code: The source code being analyzed
            queries: The list of queries (the last one will be modified to extract path data)

        Returns:
            Tuple containing success flag and extracted paths
        """
        # Modify the last query to extract JSON-formatted path information
        reachability_query = queries[-1]
        if reachability_query.endswith(".l"):
            # Remove last '.l' execution directive
            reachability_query = "".join(reachability_query.rsplit(".l", 1))

        # Extract node information for each path element
        reachability_query = reachability_query + ".map(flow => flow.elements.map(node => Map(\"id\" -> node.id, \"line_number\" -> node.lineNumber))).toJsonPretty"

        status, joern_paths = self.run_query(reachability_query)
        if status != QueryStatus.SUCCESSFUL:
            print("Joern paths query failed with the following output: ", joern_paths)
            return False, []

        print("Joern generated paths: ", joern_paths)

        # Parse the output
        try:
            # Remove first line (result declaration)
            if len(joern_paths.split("\n", 1)) != 2:
                print("Joern returned an invalid paths output:", joern_paths)
                return True, []
            joern_paths = joern_paths.split("\n", 1)[1]

            # Remove last line (closing brace)
            if len(joern_paths.rsplit("\n", 2)) != 3:
                print("Joern returned an invalid paths output (first line removed):", joern_paths)
                return True, []
            joern_paths = joern_paths.rsplit("\n", 2)[0]
            joern_paths = "[" + joern_paths + "]"

            # Parse JSON
            joern_paths_json = json.loads(joern_paths)
        except Exception as e:
            print(f"Failed to load joern output:\n{joern_paths}\nWith the error: {e}")
            return True, []

        # Extract path information with source code
        source_code_lines = source_code.splitlines()
        paths = []
        for path_json in joern_paths_json:
            path = []
            for element in path_json:
                if not str(element["line_number"]).isdigit():
                    continue
                path.append({
                    "id": element["id"],
                    "line_number": element["line_number"],
                    "line_code": source_code_lines[element["line_number"]-1]
                })
            paths.append(path)
        print("final paths: \n", paths)
        return True, paths

    def extract_sources_sinks(self, source_code: str, sources_qr: str, sinks_qr: str) -> Tuple[bool, list, list]:
        """
        Extract sources and sinks using provided queries

        Args:
            source_code: The source code being analyzed
            sources_qr: The query to identify sources
            sinks_qr: The query to identify sinks

        Returns:
            Tuple containing success flag, sources list, and sinks list
        """
        # Modify queries to extract JSON data
        sources_qr = sources_qr + ".map(node => Map(\"id\" -> node.id, \"line_number\" -> node.lineNumber)).toJsonPretty"
        sinks_qr = sinks_qr + ".map(node => Map(\"id\" -> node.id, \"line_number\" -> node.lineNumber)).toJsonPretty"

        # Run queries
        srcs_status, joern_sources = self.run_query(sources_qr)
        sinks_status, joern_sinks = self.run_query(sinks_qr)

        if srcs_status == QueryStatus.ERROR or sinks_status == QueryStatus.ERROR:
            print(f"Joern sources or sinks query failed:\nSources Output: {joern_sources}\nSinks Output: {joern_sinks}")
            return False, [], []

        # Parse sources
        try:
            # Parse sources
            joern_sources = joern_sources.split("\n", 1)[1]
            joern_sources = joern_sources.rsplit("\n", 2)[0]
            joern_sources = "[" + joern_sources + "]"

            # Parse sinks
            joern_sinks = joern_sinks.split("\n", 1)[1]
            joern_sinks = joern_sinks.rsplit("\n", 2)[0]
            joern_sinks = "[" + joern_sinks + "]"

            # Convert to JSON
            sources_json = json.loads(joern_sources)
            sinks_json = json.loads(joern_sinks)
        except Exception as e:
            print(f"Failed to parse Joern output: {e}")
            return False, [], []

        # Extract information with source code
        source_code_lines = source_code.splitlines()
        sources_list = []
        sinks_list = []

        # Process sources
        for source in sources_json:
            if not str(source["line_number"]).isdigit():
                continue
            sources_list.append({
                "id": source["id"],
                "line_number": source["line_number"],
                "line_code": source_code_lines[source["line_number"]-1]
            })

        # Process sinks
        for sink in sinks_json:
            if not str(sink["line_number"]).isdigit():
                continue
            sinks_list.append({
                "id": sink["id"],
                "line_number": sink["line_number"],
                "line_code": source_code_lines[sink["line_number"]-1]
            })

        return True, sources_list, sinks_list

    def validate_joern_paths(self, paths: list, sources: list, sinks: list, criticals: list) -> list:
        """
        Validate paths based on source, sink, and critical elements

        Args:
            paths: List of extracted paths
            sources: List of source substrings to check for
            sinks: List of sink substrings to check for
            criticals: List of critical code lines to check for

        Returns:
            List of valid paths
        """
        valid_paths = []

        for path in paths:
            isSourceExists = False
            isSinksExists = False
            isCriticalExists = False

            for element in path:
                if any(source in element["line_code"] for source in sources):
                    isSourceExists = True
                if any(sink in element["line_code"] for sink in sinks):
                    isSinksExists = True
                if len(criticals) == 0 or any(critical in element["line_code"] for critical in criticals):
                    isCriticalExists = True

            if isSourceExists and isSinksExists and isCriticalExists:
                valid_paths.append(path)

        return valid_paths

    def get_number_of_flows(self, queries: list) -> int:
        """
        Get the number of flows from a query result

        Args:
            queries: List of queries to execute

        Returns:
            Number of flows found
        """
        # Run all queries except the last one
        for qr in queries[:-1]:
            self.run_query(qr)

        # Get the size of the result
        flows_size_query = queries[-1] + ".toList.size"
        _, size_joern = self.run_query(flows_size_query)

        try:
            size = int(size_joern.split(" ")[-1])
            return size
        except Exception:
            return 0
