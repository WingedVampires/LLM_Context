# analyzer.py
import json
import os
import sys
import shutil
from .knowledge_graph import KnowledgeGraph # 从之前重构的 KG 导入
from .language_factory import LanguageConfigFactory, ParserFactory, language_by_extension, EXT_LANG_MAP
from .config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NORMAL_CONNECTION,
    STRONG_CONNECTION,
)
from datetime import datetime
class CodeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.language = config.get('language', 'python') # 默认 Python
        self.language_config = LanguageConfigFactory.get_config(self.language)
        self.parser = ParserFactory.create_parser(self.language)
        self.repo_path = config['repo_path']
        self.kg = KnowledgeGraph(
            NEO4J_URI,
            NEO4J_USER,
            NEO4J_PASSWORD,
            config.get('db_name', 'repo_analysis')# 固定数据库名，或从 config 取
        )
        self.kg.clear_graph()
        self.kg._create_indexes()
        self.processed_files = set()
    def _parser_for_file(self, file_path: str):
        lang = language_by_extension(file_path)
        if not lang:
            return None
        return ParserFactory.create_parser(lang)
    def _build_file_class_methods(self, file_path):
        parser = self._parser_for_file(file_path)
        if parser is None:
            return
        if not any(file_path.endswith(ext) for ext in self.language_config.config['file_extensions']):
            return
        import re
        if re.search(r'test[_\w]*\.(py|java|cpp)', file_path.lower()) and not file_path.endswith(self.language_config.config.get('test_file_pattern', '')):
            print(f"Skip processing test file: {file_path}")
            return
        if file_path in self.processed_files:
            print(f"File {file_path} already processed, skipping")
            return
        print(f"Processing file: {file_path}")
        self.processed_files.add(file_path)
        classes = parser.extract_classes(file_path)
        for class_info in classes:
            class_name = class_info['name'] if class_info['name'] else '__'
            self.kg.create_class_entity(
                class_name,
                class_info['file_path'],
                class_info['start_line'],
                class_info['end_line'],
                class_info.get('source_code', '') or '',
                class_info.get('doc_string', '') or '',
                STRONG_CONNECTION
            )
            self.kg.link_class_to_file(class_name, class_info['file_path'], STRONG_CONNECTION)
            for method in class_info.get('methods', []):
                method_name = method['name']
                self.kg.create_method_entity(
                    method_name,
                    method['signature'],
                    method['file_path'],
                    method['start_line'],
                    method['end_line'],
                    method['source_code'] or '',
                    method.get('doc_string', '') or '',
                    STRONG_CONNECTION
                )
                self.kg.link_class_to_method(
                    class_name,
                    class_info['file_path'],
                    method_name,
                    method['signature'],
                    STRONG_CONNECTION
                )
    def _scan_project_for_related_methods(self):
        """Scan project to find methods with call relationships"""
        print("Starting scanning project to find related methods...")
        all_supported_extensions = list(EXT_LANG_MAP.keys())
        source_files = []
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'build', 'dist']]
            for file in files:
                if any(file.endswith(ext) for ext in all_supported_extensions):
                    source_files.append(os.path.join(root, file))
        total_files = len(source_files)
        all_methods = [] # 收集所有方法，用于调用分析
        for idx, file_path in enumerate(source_files, 1):
            parser = self._parser_for_file(file_path)
            if parser is None:
                continue
            print(f"\nScanning for method calls in [{idx}/{total_files}]: {file_path}")
            self._build_file_class_methods(file_path) # 先构建实体
            # 收集本地方法
            imports = parser.get_imports(file_path)
            global_methods = parser.get_global_methods(file_path, self.repo_path)
            all_methods.extend(global_methods)
            classes = parser.extract_classes(file_path)
            for class_info in classes:
                all_methods.extend(class_info.get('methods', []))
        # 建立调用关系
        for local_method_info in all_methods:
            parser.analyze_method_calls_in_method(
                local_method_info,
                all_methods, # 使用收集的所有方法
                self.kg,
                imports,
                self.repo_path
            )
    def analyze(self):
        """Execute complete analysis flow for local repo"""
        try:
            print('======> Creating directory structure')
            self.kg.create_directory_structure(self.repo_path, self, process_detail=True, weight=STRONG_CONNECTION)
            print('======> Building file/class/method entities')
            # 依赖 create_directory_structure 中的处理，无需重复遍历
            print("======> Scanning for method calls")
            self._scan_project_for_related_methods()
            # 统计实体（简单 Cypher 查询）
            with self.kg.driver.session() as session:
                stats = {
                    'files': session.run("MATCH (f:File) RETURN count(f) as count").single()['count'],
                    'classes': session.run("MATCH (c:Class) RETURN count(c) as count").single()['count'],
                    'methods': session.run("MATCH (m:Method) RETURN count(m) as count").single()['count'],
                    'call_rels': session.run("MATCH (m1:Method)-[:RELATED {description: 'calls method'}]->(m2:Method) RETURN count(*) as count").single()['count'],
                }
            print("Analysis statistics:", stats)
            return {'entity_stats': stats}
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
        finally:
            self._cleanup()
    def _cleanup(self):
        """Clean up resources"""
        try:
            print("Cleaning temporary files...")
            # 纯 Python 清理：删除可能生成的临时文件（自定义，根据 parser）
            temp_dirs = ['__pycache__', '.pytest_cache', 'target'] # 示例，根据需要扩展，添加 Java 的 target
            for root, dirs, files in os.walk(self.repo_path):
                for temp_dir in temp_dirs:
                    temp_path = os.path.join(root, temp_dir)
                    if os.path.exists(temp_path):
                        print(f"Removing {temp_path}")
                        shutil.rmtree(temp_path)
            print("Temporary files cleaned")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        self.kg.close()
        print('Cleanup completed')
        
        
def build_graph(repo_path = '/home/hreyulog/codebase/LLM_Context/Context_KG/repos/GDsmith', language = 'java', output_dir = '../output', collection_name='test_col'):
    start_time = datetime.now()
    print(f"Starting execution: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # repo_path = sys.argv[1]
    # output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    # language = sys.argv[3] if len(sys.argv) > 3 else 'python'
    os.makedirs(output_dir, exist_ok=True)
    config = {
        'repo_path': repo_path,
        'language': language,
        'db_name': collection_name
    }
    print(f"Configuration: {config}")
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze()
    if result is None:
        print("Analysis returned no result. Exiting.")
        sys.exit(1)
    instance_id = os.path.basename(repo_path) # 用 repo 名作为 ID
    output_file = os.path.join(output_dir, f"{instance_id}_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to: {output_file}")
    end_time = datetime.now()
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"Total duration: {duration}")
    
if __name__ == "__main__":
    # if len(sys.argv) < 2:
    # print('[USAGE] python analyzer.py <repo_path> [output_dir] [language]')
    # print(' repo_path: Local repo root directory')
    # print(' output_dir: Optional, default ./output')
    # print(' language: Optional, default python (e.g., java)')
    # sys.exit(1)
    start_time = datetime.now()
    print(f"Starting execution: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # repo_path = sys.argv[1]
    # output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    # language = sys.argv[3] if len(sys.argv) > 3 else 'python'
    repo_path = '/home/hreyulog/codebase/LLM_Context/Context_KG/repos/GDsmith'
    output_dir = '../output'
    language = 'java'
    os.makedirs(output_dir, exist_ok=True)
    config = {
        'repo_path': repo_path,
        'language': language,
    }
    print(f"Configuration: {config}")
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze()
    if result is None:
        print("Analysis returned no result. Exiting.")
        sys.exit(1)
    instance_id = os.path.basename(repo_path) # 用 repo 名作为 ID
    output_file = os.path.join(output_dir, f"{instance_id}_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to: {output_file}")
    end_time = datetime.now()
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"Total duration: {duration}")
