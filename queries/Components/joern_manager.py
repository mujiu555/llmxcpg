from enum import Enum
import json
import re
import time
import subprocess
import shlex
from typing import List, Dict, Tuple, Any, Optional
from cpgqls_client import CPGQLSClient, import_code_query, delete_query


class QueryStatus(Enum):
    SUCCESSFUL = 1
    EMPTYRESULT = 2
    ERROR = 3


class JoernManager:
    """
    Manages all interactions with Joern server, including:
    - Server health monitoring and recreation
    - Running queries
    - Loading and deleting projects
    - Extracting paths and other data from Joern
    """
    
    def __init__(
        self,
        port: int,
        compose_file: Optional[str] = None,
        runtime_mode: str = "direct",
        host: str = "localhost",
        restart_command: Optional[str] = None,
    ):
        """
        Initialize a Joern Manager for a specific port
        
        Args:
            port: Joern server port number
            compose_file: Docker compose file path for server recreation.
            runtime_mode: Runtime strategy for Joern management. One of "direct" or "docker".
            host: Hostname where Joern CPGQL server is reachable.
            restart_command: Optional shell command used in direct mode to restart Joern.
                Supports "{port}" template.
        """
        self.port = port
        self.compose_file = compose_file
        self.runtime_mode = runtime_mode
        self.host = host
        self.restart_command = restart_command
        self.server_name = f"joern_server_{port}"
        self.joern_client = CPGQLSClient(f"{host}:{port}")
        
    def check_server_health(self) -> bool:
        """
        Check if the Joern server is healthy by running a simple query
        
        Returns:
            Boolean indicating if server is healthy
        """
        try:
            status, _ = self.run_query("val x = 1")
            return status == QueryStatus.SUCCESSFUL
        except Exception:
            return False
    
    def recreate_server(self) -> bool:
        """
        Recreate the Joern server to address memory issues
        
        Returns:
            Boolean indicating if server recreation was successful
        """
        try:
            # Initial small delay to prevent overwhelming the system
            time.sleep(2)
            print(f"Starting recreation of server: {self.server_name}")

            if self.runtime_mode == "docker":
                if not self.compose_file:
                    print("Docker mode requested but compose_file is not provided.")
                    return False
                # Force recreate the specific service.
                subprocess.run([
                    'docker', 'compose',
                    '-f', self.compose_file,
                    'up', '-d',
                    '--force-recreate',
                    self.server_name
                ], check=True)
            elif self.runtime_mode == "direct":
                if self.restart_command:
                    rendered_cmd = self.restart_command.format(port=self.port)
                    cmd_parts = shlex.split(rendered_cmd)
                    subprocess.run(cmd_parts, check=True)
                else:
                    print("Direct mode selected without restart command; checking current server health only.")
            else:
                print(f"Unsupported runtime mode: {self.runtime_mode}")
                return False

            # Wait for service to be fully operational
            is_healthy = self._wait_for_server_health()
            
            if is_healthy:
                print(f"Server {self.server_name} recreated and verified successfully")
            else:
                print(f"Server {self.server_name} may not be fully operational")
            
            return is_healthy
        except subprocess.CalledProcessError as e:
            print(f"Error recreating service {self.server_name}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error with service {self.server_name}: {e}")
            return False

    def _wait_for_server_health(self, max_wait: int = 180, check_interval: int = 20) -> bool:
        """
        Wait for a service to be fully operational
        
        Args:
            max_wait: Maximum time to wait (in seconds)
            check_interval: Time between health checks (in seconds)
        
        Returns:
            Boolean indicating if service became healthy
        """
        total_waited = 0
        
        while total_waited < max_wait:
            try:
                # Run query to check if the Joern server is ready to accept queries
                if self.check_server_health():
                    print(f"Joern server {self.server_name} is ready to accept requests.")
                    return True
                    
                # Wait before next check
                print(f"Waiting before next check for server: {self.server_name}")
                    
                time.sleep(check_interval)
                total_waited += check_interval
            
            except Exception as e:
                print(f"Error checking health for {self.server_name}: {e}")
                time.sleep(check_interval)
                total_waited += check_interval
        
        print(f"Server {self.server_name} did not become healthy within {max_wait} seconds")
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
    
    def load_project(self, folder_path: str) -> str:
        """
        Load a project into Joern
        
        Args:
            folder_path: Path to the folder containing the code to analyze
            
        Returns:
            Output from the import operation
        """
        import_code_qr = import_code_query(folder_path, folder_path)
        print("import query: ", import_code_qr)
        status, stdout = self.run_query(import_code_qr)
        print("stdout: ", stdout)
        print(f"Project loaded from {folder_path}")
        return stdout

    def delete_project(self, project_name: str) -> str:
        """
        Delete a project from Joern
        
        Args:
            project_name: Name of the project to delete
            
        Returns:
            Output from the delete operation
        """
        delete_project_query = delete_query(project_name)
        print("delete query: ", delete_project_query)
        status, stdout = self.run_query(delete_project_query)
        print("stdout: ", stdout)
        print(f"Project {project_name} deleted")
        return stdout
    
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
