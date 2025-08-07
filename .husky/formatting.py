#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import os
import sys
import chardet
from pathlib import Path

def get_staged_files():
    """获取暂存区中指定后缀的文件"""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=d'],
        capture_output=True, text=True
    )
    extensions = {'.c', '.cc','cpp', '.cu', '.h', '.hpp', '.cuh'}
    return [
        f for f in result.stdout.splitlines() 
        if Path(f).suffix in extensions and os.path.exists(f)
        and 'dependencies' not in f
    ]

def convert_to_utf8_bom(filepath):
    """转换文件编码为UTF-8 with BOM"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            detected = chardet.detect(content)
        f.close()

        if detected['encoding'] and detected['confidence'] > 0.9:
            if detected['encoding'].lower() not in {'utf-8-sig'}:
                with open(filepath, 'r', encoding=detected['encoding'].lower()) as f:
                    content = f.read()
                f.close()

                with open(filepath, 'w', encoding='utf-8-sig') as f:
                    f.write(content)
                f.close()
                print(f"Converting {filepath} from {detected['encoding']} with confidence {detected['confidence']} to UTF-8 with BOM")
                return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return False

def clang_format(filepath):
    """使用.clang-format进行代码格式化"""
    try:
        original_content = Path(filepath).read_text(encoding='utf-8')
        subprocess.run(
            ['clang-format','-style=file', '-i', filepath],
            check=True
        )
        formatted_content = Path(filepath).read_text(encoding='utf-8')
        return original_content != formatted_content
    except subprocess.CalledProcessError as e:
        print(f"clang-format error: {str(e)}")
        return False

if __name__ == '__main__':
    # 处理编码转换
    staged_files = get_staged_files()
    encoding_changed = set()
    
    for file in staged_files:
        if convert_to_utf8_bom(file):
            encoding_changed.add(file)
            subprocess.run(['git', 'add', file])
    
    # 处理代码格式化
    format_changed = set()
    for file in staged_files:
        if clang_format(file):
            format_changed.add(file)
            print(f"Formatting {file} with clang-format")
            subprocess.run(['git', 'add', file])
    
    # 如果有修改则返回非零退出码
    if encoding_changed or format_changed:
        print("以下文件被修改并重新暂存，请检查后重新commit:")
        print("\n".join(encoding_changed.union(format_changed)))
        sys.exit(1)

    print("所有文件已符合编码规范和代码风格，无需修改。")
    sys.exit(0)