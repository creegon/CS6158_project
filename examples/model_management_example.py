"""
模型管理示例
演示如何使用智能模型切换功能
"""
from utils import (
    switch_provider, 
    switch_model,
    show_current_config,
    show_all_models,
    list_models_by_family,
    get_available_models
)


def example_show_current():
    """示例: 显示当前配置"""
    print("\n" + "="*70)
    print("示例 1: 显示当前配置")
    print("="*70)
    
    show_current_config()


def example_list_all_models():
    """示例: 列出所有模型"""
    print("\n" + "="*70)
    print("示例 2: 列出所有可用模型")
    print("="*70)
    
    # 显示所有提供商的模型
    show_all_models()
    
    # 只显示特定提供商
    print("\n只显示 SiliconFlow 的模型:")
    show_all_models(provider='siliconflow')
    
    # 搜索特定模型
    print("\n搜索包含 'llama' 的模型:")
    show_all_models(search='llama')


def example_list_by_family():
    """示例: 按系列分组显示"""
    print("\n" + "="*70)
    print("示例 3: 按模型系列分组显示")
    print("="*70)
    
    list_models_by_family('siliconflow')


def example_switch_provider_only():
    """示例: 只切换提供商（使用默认模型）"""
    print("\n" + "="*70)
    print("示例 4: 切换到 DeepSeek (使用默认模型)")
    print("="*70)
    
    switch_provider('deepseek')


def example_switch_with_model():
    """示例: 同时切换提供商和模型"""
    print("\n" + "="*70)
    print("示例 5: 切换到 SiliconFlow 并指定模型")
    print("="*70)
    
    # 切换到 SiliconFlow 的 Llama 3.1 70B 模型
    switch_provider('siliconflow', model='meta-llama/Meta-Llama-3.1-70B-Instruct')


def example_switch_model_only():
    """示例: 只切换模型（不改变提供商）"""
    print("\n" + "="*70)
    print("示例 6: 只切换模型")
    print("="*70)
    
    # 切换到另一个模型（保持当前提供商）
    switch_model('meta-llama/Meta-Llama-3.1-8B-Instruct')


def example_get_models_programmatically():
    """示例: 编程方式获取模型列表"""
    print("\n" + "="*70)
    print("示例 7: 编程方式获取模型列表")
    print("="*70)
    
    # 获取 SiliconFlow 的所有模型
    models = get_available_models('siliconflow')
    
    print(f"SiliconFlow 共有 {len(models)} 个模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")


def example_interactive_model_selection():
    """示例: 交互式选择模型"""
    print("\n" + "="*70)
    print("示例 8: 交互式模型选择")
    print("="*70)
    
    provider = 'siliconflow'
    models = get_available_models(provider)
    
    print(f"\n{provider.upper()} 可用模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    try:
        choice = int(input(f"\n请选择模型 (1-{len(models)}): "))
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            switch_model(selected_model, provider)
        else:
            print("❌ 无效的选择")
    except (ValueError, EOFError):
        print("❌ 输入无效或已取消")


def example_load_from_external_config():
    """示例: 从外部配置文件加载模型（如果可用）"""
    print("\n" + "="*70)
    print("示例 9: 从外部配置加载模型")
    print("="*70)
    
    from utils.provider_manager import load_external_models
    
    # 尝试从 MaiBot 配置文件加载
    external_models = load_external_models()
    
    if external_models:
        print("✅ 成功从外部配置加载模型:")
        for provider, models in external_models.items():
            print(f"\n{provider.upper()}: {len(models)} 个模型")
            for model in models[:5]:  # 显示前5个
                print(f"  • {model}")
            if len(models) > 5:
                print(f"  ... 还有 {len(models) - 5} 个")
    else:
        print("ℹ️ 外部配置文件不可用，使用内置模型列表")


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                      模型管理功能演示                              ║
╚══════════════════════════════════════════════════════════════════╝

可用示例:
1. example_show_current()           - 显示当前配置
2. example_list_all_models()        - 列出所有模型
3. example_list_by_family()         - 按系列分组显示
4. example_switch_provider_only()   - 只切换提供商
5. example_switch_with_model()      - 同时切换提供商和模型
6. example_switch_model_only()      - 只切换模型
7. example_get_models_programmatically() - 编程获取模型列表
8. example_interactive_model_selection() - 交互式选择模型
9. example_load_from_external_config()   - 从外部配置加载

运行示例:
    python examples/model_management_example.py
    """)
    
    # 运行基本示例
    example_show_current()
    
    # 取消下面的注释来运行其他示例
    # example_list_all_models()
    # example_list_by_family()
    # example_switch_with_model()
    # example_interactive_model_selection()
