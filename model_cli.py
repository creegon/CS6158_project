"""
模型切换CLI工具
提供便捷的命令行界面来管理和切换模型
"""
import argparse
from utils import (
    show_current_config,
    show_all_models,
    list_models_by_family,
    switch_provider,
    switch_model,
    get_available_models
)


def cmd_show():
    """显示当前配置"""
    show_current_config()


def cmd_list(args):
    """列出所有模型"""
    if args.family:
        list_models_by_family(args.provider)
    else:
        show_all_models(provider=args.provider, search=args.search)


def cmd_switch(args):
    """切换模型"""
    if args.model:
        # 指定了模型
        switch_model(args.model, args.provider)
    elif args.provider:
        # 只指定了提供商
        switch_provider(args.provider)
    else:
        print("❌ 请指定提供商 (--provider) 或模型 (--model)")


def cmd_interactive():
    """交互式选择模型"""
    print("\n" + "="*70)
    print("交互式模型选择")
    print("="*70)
    
    # 选择提供商
    print("\n可用提供商:")
    providers = ['deepseek', 'siliconflow']
    for i, p in enumerate(providers, 1):
        print(f"  {i}. {p}")
    
    try:
        provider_choice = int(input(f"\n选择提供商 (1-{len(providers)}): "))
        if not (1 <= provider_choice <= len(providers)):
            print("❌ 无效的选择")
            return
        
        provider = providers[provider_choice - 1]
        
        # 获取模型列表
        models = get_available_models(provider)
        
        print(f"\n{provider.upper()} 可用模型:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        model_choice = int(input(f"\n选择模型 (1-{len(models)}): "))
        if not (1 <= model_choice <= len(models)):
            print("❌ 无效的选择")
            return
        
        selected_model = models[model_choice - 1]
        
        # 确认
        print(f"\n即将切换到:")
        print(f"  提供商: {provider}")
        print(f"  模型: {selected_model}")
        
        confirm = input("\n确认切换? (y/n): ").lower()
        if confirm == 'y':
            switch_model(selected_model, provider)
        else:
            print("❌ 已取消")
            
    except (ValueError, KeyboardInterrupt, EOFError):
        print("\n❌ 操作已取消")


def main():
    parser = argparse.ArgumentParser(
        description='模型管理CLI工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 显示当前配置
  python model_cli.py show
  
  # 列出所有模型
  python model_cli.py list
  
  # 列出特定提供商的模型
  python model_cli.py list --provider siliconflow
  
  # 按系列分组显示
  python model_cli.py list --family
  
  # 搜索模型
  python model_cli.py list --search llama
  
  # 切换到指定模型
  python model_cli.py switch --model meta-llama/Meta-Llama-3.1-70B-Instruct
  
  # 切换提供商（使用默认模型）
  python model_cli.py switch --provider deepseek
  
  # 交互式选择模型
  python model_cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # show命令
    parser_show = subparsers.add_parser('show', help='显示当前配置')
    
    # list命令
    parser_list = subparsers.add_parser('list', help='列出可用模型')
    parser_list.add_argument('--provider', '-p', help='指定提供商')
    parser_list.add_argument('--search', '-s', help='搜索关键词')
    parser_list.add_argument('--family', '-f', action='store_true', help='按系列分组显示')
    
    # switch命令
    parser_switch = subparsers.add_parser('switch', help='切换模型或提供商')
    parser_switch.add_argument('--provider', '-p', help='提供商名称')
    parser_switch.add_argument('--model', '-m', help='模型名称')
    
    # interactive命令
    parser_interactive = subparsers.add_parser('interactive', help='交互式选择模型')
    
    args = parser.parse_args()
    
    if args.command == 'show':
        cmd_show()
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'switch':
        cmd_switch(args)
    elif args.command == 'interactive':
        cmd_interactive()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
