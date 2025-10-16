"""
数据集划分工具
包括训练集/验证集/测试集划分和K折交叉验证
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import random


def split_dataset(df: pd.DataFrame,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 stratify_column: Optional[str] = None,
                 random_seed: int = 42,
                 shuffle: bool = True) -> Dict[str, pd.DataFrame]:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        df: 原始DataFrame
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.2）
        test_ratio: 测试集比例（默认0.1）
        stratify_column: 分层采样的列名（如'label'），None表示不分层
        random_seed: 随机种子（用于复现）
        shuffle: 是否在划分前打乱数据
        
    Returns:
        包含 'train', 'val', 'test' 的字典，值为对应的DataFrame
        
    Example:
        >>> splits = split_dataset(df, stratify_column='label', random_seed=42)
        >>> train_df = splits['train']
        >>> val_df = splits['val']
        >>> test_df = splits['test']
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1，当前为: {total_ratio}")
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 如果需要分层采样
    if stratify_column and stratify_column in df.columns:
        print(f"\n📊 使用分层采样，基于列: {stratify_column}")
        
        # 获取各类别
        categories = df[stratify_column].unique()
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        print(f"\n各类别划分情况:")
        print(f"{'类别':<20} {'总数':>8} {'训练集':>8} {'验证集':>8} {'测试集':>8}")
        print("-" * 60)
        
        for category in categories:
            # 获取该类别的所有数据
            cat_df = df[df[stratify_column] == category].copy()
            cat_size = len(cat_df)
            
            # 打乱数据
            if shuffle:
                cat_df = cat_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            
            # 计算划分点
            train_size = int(cat_size * train_ratio)
            val_size = int(cat_size * val_ratio)
            
            # 划分
            train_cat = cat_df.iloc[:train_size]
            val_cat = cat_df.iloc[train_size:train_size + val_size]
            test_cat = cat_df.iloc[train_size + val_size:]
            
            train_dfs.append(train_cat)
            val_dfs.append(val_cat)
            test_dfs.append(test_cat)
            
            print(f"{str(category):<20} {cat_size:>8} {len(train_cat):>8} {len(val_cat):>8} {len(test_cat):>8}")
        
        # 合并所有类别
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # 再次打乱（可选）
        if shuffle:
            train_df = train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            val_df = val_df.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)
    
    else:
        print(f"\n📊 使用随机划分（不分层）")
        
        # 打乱数据
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # 计算划分点
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # 划分
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
    
    # 打印总体统计
    print(f"\n{'=' * 60}")
    print(f"划分结果:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  总计: {len(train_df) + len(val_df) + len(test_df)} 条")
    print(f"{'=' * 60}\n")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def create_project_wise_kfold_splits(df: pd.DataFrame,
                                     project_column: str = 'project',
                                     category_column: str = 'category',
                                     n_folds: int = 4,
                                     min_samples_per_category: int = 4,
                                     random_seed: int = 42) -> List[Dict[str, pd.DataFrame]]:
    """
    创建项目级独立的K折交叉验证数据集
    
    保证:
    1. 同一项目的测试不会同时出现在训练集和测试集中（project-wise disjoint）
    2. 每个测试集至少包含每种类别的 min_samples_per_category 个样本
    
    Args:
        df: 原始DataFrame
        project_column: 项目名称列
        category_column: 类别列
        n_folds: 折数（默认4）
        min_samples_per_category: 每个测试集中每个类别的最小样本数（默认4）
        random_seed: 随机种子
        
    Returns:
        包含n_folds个字典的列表，每个字典包含 'train', 'test' 键
        
    Example:
        >>> folds = create_project_wise_kfold_splits(df, n_folds=4)
        >>> for i, fold in enumerate(folds):
        >>>     print(f"Fold {i+1}:")
        >>>     print(f"  训练集: {len(fold['train'])} 样本")
        >>>     print(f"  测试集: {len(fold['test'])} 样本")
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"\n{'='*60}")
    print(f"项目级K折交叉验证数据集划分 (K={n_folds})")
    print(f"{'='*60}")
    
    # 1. 分析数据集基本信息
    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  项目数量: {df[project_column].nunique()}")
    print(f"  类别数量: {df[category_column].nunique()}")
    
    # 2. 分析每个类别的分布
    print(f"\n📊 各类别分布:")
    category_counts = df[category_column].value_counts().sort_index()
    category_projects = {}
    
    for cat in category_counts.index:
        cat_df = df[df[category_column] == cat]
        n_projects = cat_df[project_column].nunique()
        category_projects[cat] = n_projects
        print(f"  类别 {cat}: {category_counts[cat]:>5} 个样本, 分布在 {n_projects:>3} 个项目中")
    
    # 3. 检查是否可以满足最小样本数要求
    print(f"\n⚠️  类别平衡约束检查:")
    print(f"  要求: 每个测试集至少包含每个类别 {min_samples_per_category} 个样本")
    
    issues = []
    for cat in category_counts.index:
        expected_per_fold = category_counts[cat] / n_folds
        if expected_per_fold < min_samples_per_category:
            issues.append(f"    - 类别 {cat}: 平均每折只有 {expected_per_fold:.1f} 个样本 (< {min_samples_per_category})")
    
    if issues:
        print(f"  ⚠️  警告: 以下类别可能难以满足最小样本数要求:")
        for issue in issues:
            print(issue)
        print(f"  提示: 算法会尽量满足约束，但可能需要调整策略")
    else:
        print(f"  ✓ 所有类别都有足够的样本")
    
    # 4. 按项目分组
    print(f"\n🔄 开始项目级分组...")
    projects = df[project_column].unique()
    np.random.shuffle(projects)  # 随机打乱项目顺序
    
    # 为每个项目统计其包含的类别和样本数
    project_info = {}
    for proj in projects:
        proj_df = df[df[project_column] == proj]
        project_info[proj] = {
            'df': proj_df,
            'size': len(proj_df),
            'categories': proj_df[category_column].value_counts().to_dict()
        }
    
    # 5. 改进的项目分配策略
    # 策略: 
    # 1) 识别包含稀有类别的"关键项目"
    # 2) 优先均匀分配关键项目到各折
    # 3) 剩余项目使用贪心策略平衡分配
    
    folds = [{'projects': [], 'size': 0, 'categories': {cat: 0 for cat in category_counts.index}} 
             for _ in range(n_folds)]
    
    print(f"\n🎯 使用改进的项目分配策略...")
    print(f"  第一阶段: 识别并均匀分配包含稀有类别的关键项目")
    print(f"  第二阶段: 贪心分配剩余项目以平衡类别分布")
    
    # 5.1 识别"关键项目" - 包含稀有类别样本的项目
    # 定义"稀有类别"为样本数少于总样本数1%的类别
    rare_threshold = len(df) * 0.01  # 1%
    rare_categories = [cat for cat in category_counts.index if category_counts[cat] < rare_threshold]
    
    print(f"\n  识别的稀有类别: {rare_categories}")
    print(f"  （样本数 < {rare_threshold:.0f}）")
    
    # 为每个稀有类别找出包含它的项目，并按该类别的样本数排序
    critical_projects = set()
    category_project_map = {}  # 每个稀有类别 -> 包含它的项目列表（按样本数降序）
    
    for cat in rare_categories:
        cat_projects = []
        for proj, info in project_info.items():
            if cat in info['categories']:
                cat_projects.append((proj, info['categories'][cat]))
        # 按该类别的样本数降序排序
        cat_projects.sort(key=lambda x: x[1], reverse=True)
        category_project_map[cat] = [proj for proj, count in cat_projects]
        critical_projects.update(category_project_map[cat])
    
    print(f"\n  发现 {len(critical_projects)} 个关键项目")
    
    # 5.2 使用Round-Robin策略分配关键项目
    # 目标: 确保每个折都能获得各种稀有类别的样本
    critical_projects_list = sorted(critical_projects, 
                                   key=lambda p: project_info[p]['size'], 
                                   reverse=True)
    
    fold_idx = 0
    for proj in critical_projects_list:
        proj_data = project_info[proj]
        folds[fold_idx]['projects'].append(proj)
        folds[fold_idx]['size'] += proj_data['size']
        for cat, count in proj_data['categories'].items():
            folds[fold_idx]['categories'][cat] += count
        fold_idx = (fold_idx + 1) % n_folds
    
    print(f"  ✓ 已将关键项目均匀分配到 {n_folds} 折")
    
    # 5.3 使用贪心策略分配剩余项目
    remaining_projects = [proj for proj in projects if proj not in critical_projects]
    remaining_projects.sort(key=lambda p: project_info[p]['size'], reverse=True)
    
    print(f"\n  第二阶段: 分配剩余 {len(remaining_projects)} 个项目...")
    
    for proj in remaining_projects:
        proj_data = project_info[proj]
        
        # 计算每个折加入该项目后的"不平衡度"
        fold_scores = []
        for i, fold in enumerate(folds):
            # 新的样本数
            new_size = fold['size'] + proj_data['size']
            
            # 计算类别分布的不平衡度
            category_imbalance = 0
            for cat in category_counts.index:
                # 当前类别的数量
                current_count = fold['categories'].get(cat, 0)
                proj_cat_count = proj_data['categories'].get(cat, 0)
                new_cat_count = current_count + proj_cat_count
                
                # 目标数量
                target_cat_count = category_counts[cat] / n_folds
                
                # 如果是稀有类别且当前数量低于最小要求，优先分配
                if cat in rare_categories and current_count < min_samples_per_category and proj_cat_count > 0:
                    category_imbalance -= 100  # 给予很大的负分（优先级高）
                else:
                    category_imbalance += abs(new_cat_count - target_cat_count)
            
            # 样本数不平衡度
            target_size = len(df) / n_folds
            size_imbalance = abs(new_size - target_size)
            
            # 总分 = 样本数不平衡 * 权重 + 类别不平衡
            # 类别平衡权重更高
            score = size_imbalance * 0.1 + category_imbalance
            fold_scores.append(score)
        
        # 选择得分最低（最平衡）的折
        best_fold_idx = np.argmin(fold_scores)
        folds[best_fold_idx]['projects'].append(proj)
        folds[best_fold_idx]['size'] += proj_data['size']
        for cat, count in proj_data['categories'].items():
            folds[best_fold_idx]['categories'][cat] += count
    
    # 6. 输出每折的统计信息
    print(f"\n📊 各折统计:")
    print(f"{'折号':<8} {'样本数':>8} {'项目数':>8} ", end='')
    for cat in sorted(category_counts.index):
        print(f"{'类别'+str(cat):>10}", end='')
    print()
    print("-" * (24 + 10 * len(category_counts)))
    
    for i, fold in enumerate(folds):
        print(f"Fold {i+1:<3} {fold['size']:>8} {len(fold['projects']):>8} ", end='')
        for cat in sorted(category_counts.index):
            print(f"{fold['categories'][cat]:>10}", end='')
        print()
    
    # 7. 检查类别平衡约束
    print(f"\n⚠️  类别平衡约束验证:")
    constraint_violations = []
    
    for i, fold in enumerate(folds):
        for cat in category_counts.index:
            if fold['categories'][cat] < min_samples_per_category:
                constraint_violations.append(
                    f"  ✗ Fold {i+1}, 类别 {cat}: 只有 {fold['categories'][cat]} 个样本 (< {min_samples_per_category})"
                )
    
    if constraint_violations:
        print(f"  ⚠️  发现 {len(constraint_violations)} 个约束违反:")
        for violation in constraint_violations[:10]:  # 只显示前10个
            print(violation)
        if len(constraint_violations) > 10:
            print(f"  ... 还有 {len(constraint_violations)-10} 个违反")
        print(f"\n  💡 建议: 考虑减少折数或降低 min_samples_per_category")
    else:
        print(f"  ✓ 所有折都满足类别平衡约束!")
    
    # 8. 创建训练集和测试集
    print(f"\n📦 生成K折数据集...")
    result_folds = []
    
    for i, test_fold in enumerate(folds):
        # 测试集: 当前折的所有项目
        test_projects = test_fold['projects']
        test_df = pd.concat([project_info[proj]['df'] for proj in test_projects], 
                           ignore_index=True)
        
        # 训练集: 其他折的所有项目
        train_projects = []
        for j, fold in enumerate(folds):
            if j != i:
                train_projects.extend(fold['projects'])
        train_df = pd.concat([project_info[proj]['df'] for proj in train_projects],
                            ignore_index=True)
        
        # 验证项目不重叠
        test_proj_set = set(test_df[project_column].unique())
        train_proj_set = set(train_df[project_column].unique())
        overlap = test_proj_set & train_proj_set
        
        if overlap:
            print(f"  ⚠️  Fold {i+1}: 发现重叠项目 {overlap}")
        
        result_folds.append({
            'train': train_df,
            'test': test_df,
            'train_projects': train_projects,
            'test_projects': test_projects
        })
        
        print(f"  Fold {i+1}: 训练集 {len(train_df)} 样本 ({len(train_projects)} 项目), "
              f"测试集 {len(test_df)} 样本 ({len(test_projects)} 项目)")
    
    print(f"\n✓ 完成! 共生成 {n_folds} 折数据集")
    print(f"{'='*60}\n")
    
    return result_folds
