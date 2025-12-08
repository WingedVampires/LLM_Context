# knowledge_graph.py
from neo4j import GraphDatabase
import os
from embedding import Embedding
from utils import relative_path
from config import (
    VECTOR_SIMILARITY_WEIGHT,
)
class KnowledgeGraph:
    def __init__(self, uri, user, password, database_name):
        database_name = database_name.replace('-', '').replace('_', '')
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            self.embedder = Embedding()
            print("KnowledgeGraph: Embedding instance created successfully")
        except Exception as e:
            print(f"KnowledgeGraph: Embedding instance creation failed: {e}")
            raise
   
    def close(self):
        self.driver.close()
   
    def _get_embedding(self, text):
        return self.embedder.get_embedding(text)
    def create_method_entity(self, method_name, method_signature, file_path, start_line, end_line, source_code, doc_string='', weight=1):
        source_code = source_code or ''
        doc_string = doc_string or ''
        # First check if method already exists using full composite key
        with self.driver.session() as session:
            exists_query = """
            MATCH (m:Method {name: $name, signature: $signature, file_path: $file_path})
            RETURN count(m) > 0 as exists
            """
            exists = session.run(exists_query,
                               name=method_name,
                               signature=method_signature,
                               file_path=file_path).single()['exists']
           
            if not exists:
                # If method doesn't exist, calculate embedding and create new method
                text_for_embedding = f"{method_name}\n{doc_string}\n{source_code}"
                if len(text_for_embedding) > 4000:
                    print(f"Warning: Truncating embedding text for method {method_name} from {len(text_for_embedding)} to 4000 chars")
                    text_for_embedding = text_for_embedding[:4000]
                embedding = self._get_embedding(text_for_embedding)
               
                session.execute_write(self._create_and_link,
                                    method_name,
                                    method_signature,
                                    file_path,
                                    start_line,
                                    end_line,
                                    source_code,
                                    doc_string,
                                    embedding,
                                    weight)
                # Create method-file relationship
                session.execute_write(self._link_method_to_file,
                                    method_name,
                                    method_signature,
                                    file_path,
                                    weight)
            else:
                # Update existing method properties
                session.execute_write(self._update_method_properties,
                                    method_name,
                                    method_signature,
                                    file_path,
                                    start_line,
                                    end_line,
                                    source_code,
                                    doc_string,
                                    weight)

    @staticmethod
    def _create_and_link(tx, method_name, method_signature, file_path, start_line, end_line, source_code, doc_string, embedding, weight):
        query = (
            "MERGE (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "ON CREATE SET m.start_line = $start_line, m.end_line = $end_line, m.source_code = $source_code, "
            "m.doc_string = $doc_string, m.embedding = $embedding "
            "ON MATCH SET m.start_line = $start_line, m.end_line = $end_line, m.source_code = $source_code, "
            "m.doc_string = $doc_string, m.embedding = $embedding"
            # Removed direct linking to file here, as it's handled by a separate call in create_method_entity
            # "MERGE (m)-[:RELATED {description: 'contained in file', weight: $weight}]->(f)"
            # "MERGE (f)-[:RELATED {description: 'contains method', weight: $weight}]->(m)"
        )
        tx.run(query, method_name=method_name, method_signature=method_signature,
               file_path=file_path, start_line=start_line, end_line=end_line, source_code=source_code, doc_string=doc_string, embedding=embedding, weight=weight)

    @staticmethod
    def _update_method_properties(tx, method_name, method_signature, file_path, start_line, end_line, source_code, doc_string, weight):
        query = (
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "SET m.start_line = $start_line, m.end_line = $end_line, m.source_code = $source_code, "
            "m.doc_string = $doc_string"
        )
        tx.run(query, method_name=method_name, method_signature=method_signature,
               file_path=file_path, start_line=start_line, end_line=end_line, source_code=source_code, doc_string=doc_string, weight=weight)

    def clear_graph(self):
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
           
            try:
                # Delete all indexes
                for index in session.run("SHOW INDEXES"):
                    session.run(f"DROP INDEX {index['name']}")
            except Exception as e:
                print(f"Error deleting indexes: {e}")
           
            try:
                # Delete all constraints
                for constraint in session.run("SHOW CONSTRAINTS"):
                    session.run(f"DROP CONSTRAINT {constraint['name']}")
            except Exception as e:
                print(f"Error deleting constraints: {e}")
    def create_file_entity(self, file_path):
        """
        Create code file entity
       
        Args:
            file_path (str): File path
        """
        with self.driver.session() as session:
            session.execute_write(self._create_file, file_path)
    @staticmethod
    def _create_file(tx, file_path):
        rel_path = relative_path(file_path)
        name = os.path.basename(rel_path) if rel_path else rel_path
        query = (
            "MERGE (f:File {path: $file_path}) "
            "ON CREATE SET f.name = $name "
            "ON MATCH SET f.name = $name"
        )
        tx.run(query, file_path=rel_path, name=name)
    def create_directory_structure(self, base_path, code_analyzer, process_detail=False, weight=1):
        """
        Create directory structure, including directories and files and their relationships
       
        Args:
            base_path (str): Base path
        """
        with self.driver.session() as session:
            file_paths = session.execute_write(self._create_directory_structure, base_path, weight)
            if process_detail and file_paths:
                for file_path in file_paths:
                    code_analyzer._build_file_class_methods(file_path)
    @staticmethod
    def _create_directory_structure(tx, base_path, weight=1):
        file_paths = []
        for root, dirs, files in os.walk(base_path):
            # Create current directory
            abs_dir_path = root.replace('\\', '/')
            rel_dir_path = relative_path(abs_dir_path)
            if os.path.basename(root).startswith('.'):
                continue
            # Create current directory node
            dir_name = os.path.basename(root) or '/'
            query = (
                "MERGE (d:Directory {path: $dir_path}) "
                "ON CREATE SET d.name = $name "
                "ON MATCH SET d.name = $name"
            )
            tx.run(query,
                dir_path=rel_dir_path or '/',
                name=dir_name
            )
            # If not root directory, create relationship with parent directory
            if rel_dir_path:
                parent_dir_abs = os.path.dirname(abs_dir_path)
                parent_dir_path = relative_path(parent_dir_abs)
                query = (
                    "MATCH (parent:Directory {path: $parent_path}) "
                    "MATCH (child:Directory {path: $child_path}) "
                    "MERGE (parent)-[:RELATED {description: 'contains directory', weight: $weight}]->(child)"
                    "MERGE (child)-[:RELATED {description: 'contained in directory', weight: $weight}]->(parent)"
                )
                tx.run(query,
                    parent_path=parent_dir_path or '/',
                    child_path=rel_dir_path,
                    weight=weight
                )
           
            # Process supported files
            supported_extensions = ['.py', '.cpp', '.java', '.h', '.hpp']  # Can be extended from config
            py_files = [f for f in files if any(f.endswith(ext) for ext in supported_extensions)]
            total_files = len(py_files)
            for idx, file in enumerate(py_files, 1):
                print(f'\nProcessing file [{idx}/{total_files}] ({(idx/total_files*100):.1f}%): {file}')
                file_abs_path = os.path.join(abs_dir_path, file)
                rel_file_path = relative_path(file_abs_path)
               
                # Create file node
                file_name = os.path.basename(rel_file_path)
                query = (
                    "MERGE (f:File {path: $file_path}) "
                    "ON CREATE SET f.name = $name "
                    "ON MATCH SET f.name = $name"
                )
                tx.run(query,
                    file_path=rel_file_path,
                    name=file_name
                )
               
                # Create directory-file relationship
                query = (
                    "MATCH (d:Directory {path: $dir_path}) "
                    "MATCH (f:File {path: $file_path}) "
                    "MERGE (d)-[:RELATED {description: 'contains file', weight: $weight}]->(f)"
                    "MERGE (f)-[:RELATED {description: 'contained in directory', weight: $weight}]->(d)"
                )
                file_paths.append(file_abs_path)
                tx.run(query,
                    dir_path=rel_dir_path or '/',
                    file_path=rel_file_path,
                    weight=weight
                )
        print(file_paths)
        return file_paths
    def create_class_entity(self, class_name, file_path, start_line, end_line, source_code, doc_string="", weight=1):
        source_code = source_code or ''
        doc_string = doc_string or ''
        with self.driver.session() as session:
            # First check if class already exists using full composite key
            exists_query = """
            MATCH (c:Class {name: $name, file_path: $file_path})
            RETURN count(c) > 0 as exists
            """
            exists = session.run(exists_query,
                               name=class_name,
                               file_path=file_path).single()['exists']
           
            if not exists:
                # If class doesn't exist, calculate embedding and create new class
                text_for_embedding = f"{class_name}\n{doc_string}\n{source_code}"
                if len(text_for_embedding) > 8000:
                    print(f"Warning: Truncating embedding text for class {class_name} from {len(text_for_embedding)} to 8000 chars")
                    text_for_embedding = text_for_embedding[:8000]
                embedding = self._get_embedding(text_for_embedding)
               
                session.execute_write(self._create_class,
                                    class_name,
                                    file_path,
                                    start_line,
                                    end_line,
                                    source_code,
                                    doc_string,
                                    embedding,
                                    weight)
            else:
                # Update existing class properties
                session.execute_write(self._update_class_properties,
                                    class_name,
                                    file_path,
                                    start_line,
                                    end_line,
                                    source_code,
                                    doc_string,
                                    weight)

    @staticmethod
    def _create_class(tx, class_name, file_path, start_line, end_line, source_code, doc_string="", embedding=None, weight=1):
        # Create class node
        query = (
            "MERGE (c:Class {name: $class_name, file_path: $file_path}) "
            "ON CREATE SET c.start_line = $start_line, c.end_line = $end_line, c.source_code = $source_code, "
            "c.doc_string = $doc_string, c.embedding = $embedding, c.short_name = $short_name "
            "ON MATCH SET c.start_line = $start_line, c.end_line = $end_line, c.source_code = $source_code, "
            "c.doc_string = $doc_string, c.embedding = $embedding, c.short_name = $short_name"
        )
        tx.run(query,
            class_name=class_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            doc_string=doc_string,
            embedding=embedding,
            short_name=class_name.split('.')[-1],
            weight=weight
        )
       
        # Create relationship with file
        query = (
            "MATCH (f:File {path: $file_path}) "
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
            "MERGE (f)-[:RELATED {description: 'contains class', weight: $weight}]->(c)"
            "MERGE (c)-[:RELATED {description: 'contained in file', weight: $weight}]->(f)"
        )
        tx.run(query, file_path=file_path, class_name=class_name, weight=weight)

    @staticmethod
    def _update_class_properties(tx, class_name, file_path, start_line, end_line, source_code, doc_string, weight=1):
        query = (
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
            "SET c.start_line = $start_line, c.end_line = $end_line, c.source_code = $source_code, "
            "c.doc_string = $doc_string"
        )
        tx.run(query,
            class_name=class_name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=source_code,
            doc_string=doc_string,
            weight=weight
        )

    def link_class_to_method(self, class_name, file_path, method_name, method_signature, weight=1):
        """
        Establish association between class and method
       
        Args:
            class_name (str): Class name
            file_path (str): File path
            method_name (str): Method name
            method_signature (str): Method signature
        """
        with self.driver.session() as session:
            session.execute_write(self._link_class_to_method,
                                class_name, file_path, method_name, method_signature, weight)
    @staticmethod
    def _link_class_to_method(tx, class_name, file_path, method_name, method_signature, weight=1):
        query = (
            "MATCH (c:Class {name: $class_name, file_path: $file_path}) "
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "MERGE (c)-[:RELATED {description: 'contains method', weight: $weight}]->(m)"
            "MERGE (m)-[:RELATED {description: 'contained in class', weight: $weight}]->(c)"
        )
        tx.run(query,
            class_name=class_name,
            file_path=file_path,
            method_name=method_name,
            method_signature=method_signature,
            weight=weight
        )
    @staticmethod
    def _link_method_to_file(tx, method_name, method_signature, file_path, weight=1):
        query = (
            "MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path}) "
            "MATCH (f:File {path: $file_path}) "
            "MERGE (f)-[:RELATED {description: 'contains method', weight: $weight}]->(m)"
            "MERGE (m)-[:RELATED {description: 'contained in file', weight: $weight}]->(f)"
        )
        tx.run(query,
            method_name=method_name,
            method_signature=method_signature,
            file_path=file_path,
            weight=weight
        )
    def link_method_calls(self, caller_name, caller_signature,
                         callee_name, callee_signature):
        with self.driver.session() as session:
            session.execute_write(self._link_method_calls,
                                caller_name, caller_signature,
                                callee_name, callee_signature)
    @staticmethod
    def _link_method_calls(tx, caller_name, caller_signature,
                          callee_name, callee_signature):
        query = (
            "MATCH (caller:Method {name: $caller_name, signature: $caller_signature}) "
            "MATCH (callee:Method {name: $callee_name, signature: $callee_signature}) "
            "MERGE (caller)-[r:RELATED {description: 'calls method'}]->(callee) "
            "ON CREATE SET r.weight = 1 "
            "ON MATCH SET r.weight = coalesce(r.weight, 0) + 1 "
            "MERGE (callee)-[r2:RELATED {description: 'called by method'}]->(caller) "
            "ON CREATE SET r2.weight = 1 "
            "ON MATCH SET r2.weight = coalesce(r2.weight, 0) + 1"
        )
        tx.run(query,
            caller_name=caller_name,
            caller_signature=caller_signature,
            callee_name=callee_name,
            callee_signature=callee_signature,
        )
    def get_method_by_name(self, method_name, file_path=None):
        with self.driver.session() as session:
            query = """
            MATCH (m:Method)
            WHERE m.name = $method_name
            """
            params = {'method_name': method_name}
            if file_path:
                query += " AND m.file_path = $file_path"
                params['file_path'] = file_path
            query += """
            RETURN m.name as name,
                    m.signature as signature,
                    m.file_path as file_path,
                    m.start_line as start_line,
                    m.end_line as end_line,
                    m.source_code as source_code,
                    m.doc_string as doc_string
            """
            result = session.run(query, **params)
            return [{
                'name': record['name'],
                'signature': record['signature'],
                'file_path': record['file_path'],
                'start_line': record['start_line'],
                'end_line': record['end_line'],
                'source_code': record['source_code'],
                'doc_string': record['doc_string']
            } for record in result]
    def search_file_by_path(self, file_path):
        parts = file_path.replace('\\', '/').split('/')
        if '~' in parts:
            parts = parts[parts.index('~')+1:]
        if len(parts) > 3:
            parts = parts[-4:]
        target_filename = parts[-1]
        query = """
        MATCH (f:File)
        WITH f, $file_parts as parts, $target_filename as target
        WITH f, parts, target, f.path as path,
             last(split(f.path, '/')) as file_name,
             split(f.path, '/') as path_parts
        WITH f, parts, target, path, file_name, path_parts,
             [p in parts WHERE p IN path_parts] as matched_parts,
             CASE
                WHEN file_name = target THEN 3
                WHEN file_name STARTS WITH 'test_' THEN 0
                WHEN file_name CONTAINS replace(target, '_', '') THEN 1
                ELSE 0
             END as filename_match_score,
             apoc.coll.indexOf(path_parts, last(parts[..-1])) as dir_match
        WHERE size(matched_parts) >= 1
        WITH f, matched_parts, filename_match_score,
             CASE WHEN dir_match >= 0 THEN 2 ELSE 0 END as same_dir,
             reduce(s = 0, i IN range(0, size(matched_parts)-1) |
                s + CASE WHEN apoc.coll.indexOf(path_parts, matched_parts[i]) < apoc.coll.indexOf(path_parts, matched_parts[i+1])
                    THEN 1 ELSE 0 END
             ) as consecutive_count,
             size(matched_parts) as match_count
        RETURN {
            file: f,
            match_count: match_count,
            consecutive_count: consecutive_count,
            score: same_dir * 1000 + filename_match_score * 100 + match_count * 10 + consecutive_count
        } as result
        ORDER BY result.score DESC
        LIMIT 3
        """
        with self.driver.session() as session:
            results = session.run(query, file_parts=parts, target_filename=target_filename)
            matches = []
            for record in results:
                matches.append({
                    'file': record['result']['file'],
                    'score': record['result']['score']
                })
            return matches if matches else None
    def _create_indexes(self):
        """Create database indexes to improve query performance"""
        with self.driver.session() as session:
            # Method node index and constraint
            session.run("""
                CREATE CONSTRAINT method_unique IF NOT EXISTS
                FOR (m:Method)
                REQUIRE (m.name, m.signature, m.file_path) IS UNIQUE
            """)
            session.run("""
                CREATE INDEX method_composite IF NOT EXISTS
                FOR (m:Method)
                ON (m.name, m.signature, m.file_path)
            """)
           
            # File node index
            session.run("""
                CREATE INDEX file_path IF NOT EXISTS
                FOR (f:File)
                ON (f.path)
            """)
           
            # Class node index and constraint
            session.run("""
                CREATE CONSTRAINT class_unique IF NOT EXISTS
                FOR (c:Class)
                REQUIRE (c.name, c.file_path) IS UNIQUE
            """)
            session.run("""
                CREATE INDEX class_composite IF NOT EXISTS
                FOR (c:Class)
                ON (c.name, c.file_path)
            """)
           
            # Directory node index
            session.run("""
                CREATE INDEX directory_path IF NOT EXISTS
                FOR (d:Directory)
                ON (d.path)
            """)
           
            print("Successfully created all indexes and constraints")
               
    def link_class_to_file(self, class_name, file_path, weight=1):
        """
        Establish relationship between class and file
       
        Args:
            class_name (str): Class name
            file_path (str): File path
        """
        with self.driver.session() as session:
            query = """
            MATCH (c:Class {name: $class_name, file_path: $file_path})
            MATCH (f:File {path: $file_path})
            MERGE (c)-[:RELATED {description: 'contained in file', weight: $weight}]->(f)
            MERGE (f)-[:RELATED {description: 'contains class', weight: $weight}]->(c)
            """
            session.run(query,
                class_name=class_name,
                file_path=file_path,
                weight=weight
            )
           
    def link_method_to_file(self, method_name, method_signature, file_path, weight=1):
        """
        Establish relationship between method and file
       
        Args:
            method_name (str): Method name
            method_signature (str): Method signature
            file_path (str): File path
        """
        with self.driver.session() as session:
            query = """
            MATCH (m:Method {name: $method_name, signature: $method_signature, file_path: $file_path})
            MATCH (f:File {path: $file_path})
            MERGE (m)-[:RELATED {description: 'contained in file', weight: $weight}]->(f)
            MERGE (f)-[:RELATED {description: 'contains method', weight: $weight}]->(m)
            """
            session.run(query,
                method_name=method_name,
                method_signature=method_signature,
                file_path=file_path,
                weight=weight
            )