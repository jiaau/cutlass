#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个latex文件中tikz节点坐标对应的值是否相等
"""

import re
import sys
from typing import Dict, Tuple, Set

def parse_latex_nodes(file_path: str) -> Dict[Tuple[int, int], str]:
    """
    解析latex文件中的tikz节点，提取坐标和对应的值
    
    Args:
        file_path: latex文件路径
        
    Returns:
        字典，键为(x, y)坐标元组，值为节点中的值
    """
    nodes = {}
    
    # 匹配tikz节点的正则表达式
    # \node[fill=black!XX] at (x,y) {value};
    pattern = r'\\node\[.*?\]\s+at\s+\((\d+),(\d+)\)\s+\{([^}]+)\};'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        matches = re.findall(pattern, content)
        
        for match in matches:
            x, y, value = match
            coordinate = (int(x), int(y))
            nodes[coordinate] = value.strip()
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return {}
    except Exception as e:
        print(f"错误：读取文件 {file_path} 时出现问题: {e}")
        return {}
    
    return nodes

def compare_latex_files(file1_path: str, file2_path: str) -> None:
    """
    比较两个latex文件中坐标对应的值
    
    Args:
        file1_path: 第一个latex文件路径
        file2_path: 第二个latex文件路径
    """
    print(f"正在比较文件:")
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    print("-" * 50)
    
    # 解析两个文件
    nodes1 = parse_latex_nodes(file1_path)
    nodes2 = parse_latex_nodes(file2_path)
    
    if not nodes1 or not nodes2:
        print("解析文件失败，请检查文件路径和格式")
        return
    
    print(f"文件1包含 {len(nodes1)} 个节点")
    print(f"文件2包含 {len(nodes2)} 个节点")
    print()
    
    # 获取所有坐标
    all_coords = set(nodes1.keys()) | set(nodes2.keys())
    coords_only_in_file1 = set(nodes1.keys()) - set(nodes2.keys())
    coords_only_in_file2 = set(nodes2.keys()) - set(nodes1.keys())
    common_coords = set(nodes1.keys()) & set(nodes2.keys())
    
    # 报告只在某个文件中存在的坐标
    if coords_only_in_file1:
        print(f"只在文件1中存在的坐标 ({len(coords_only_in_file1)} 个):")
        for coord in sorted(coords_only_in_file1):
            print(f"  {coord}: {nodes1[coord]}")
        print()
    
    if coords_only_in_file2:
        print(f"只在文件2中存在的坐标 ({len(coords_only_in_file2)} 个):")
        for coord in sorted(coords_only_in_file2):
            print(f"  {coord}: {nodes2[coord]}")
        print()
    
    # 比较共同坐标的值
    different_values = []
    same_values = []
    
    for coord in sorted(common_coords):
        value1 = nodes1[coord]
        value2 = nodes2[coord]
        
        if value1 != value2:
            different_values.append((coord, value1, value2))
        else:
            same_values.append((coord, value1))
    
    print(f"共同坐标比较结果:")
    print(f"  相同值的坐标: {len(same_values)} 个")
    print(f"  不同值的坐标: {len(different_values)} 个")
    print()
    
    if different_values:
        print("值不相等的坐标:")
        print("坐标\t\t文件1值\t文件2值")
        print("-" * 40)
        for coord, val1, val2 in different_values:
            print(f"{coord}\t\t{val1}\t{val2}")
        print()
    
    # 总结
    total_coords = len(all_coords)
    matching_coords = len(same_values)
    
    print("=" * 50)
    print("比较总结:")
    print(f"总坐标数: {total_coords}")
    print(f"完全匹配的坐标: {matching_coords}")
    print(f"值不同的坐标: {len(different_values)}")
    print(f"只在文件1中的坐标: {len(coords_only_in_file1)}")
    print(f"只在文件2中的坐标: {len(coords_only_in_file2)}")
    
    if len(different_values) == 0 and len(coords_only_in_file1) == 0 and len(coords_only_in_file2) == 0:
        print("\n✅ 两个文件完全一致！")
    else:
        print("\n❌ 两个文件存在差异")

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("使用方法: python compare_latex_nodes.py <文件1路径> <文件2路径>")
        print("示例: python compare_latex_nodes.py file1.tex file2.tex")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    compare_latex_files(file1_path, file2_path)

if __name__ == "__main__":
    main()
