"""
MultiAgent - 多Agent协作框架
用于协调多个Agent共同完成复杂任务
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from agents.base_agent import BaseAgent


class MultiAgentCoordinator(ABC):
    """
    多Agent协调器基类
    管理多个Agent的协作流程
    """
    
    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        """
        初始化协调器
        
        Args:
            agents: Agent列表
        """
        self.agents: List[BaseAgent] = agents or []
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: BaseAgent, name: Optional[str] = None) -> None:
        """
        添加Agent
        
        Args:
            agent: Agent实例
            name: Agent名称（可选）
        """
        if name:
            agent.name = name
        self.agents.append(agent)
        print(f"✓ 已添加Agent: {getattr(agent, 'name', type(agent).__name__)}")
    
    def remove_agent(self, index: int) -> None:
        """
        移除Agent
        
        Args:
            index: Agent索引
        """
        if 0 <= index < len(self.agents):
            removed = self.agents.pop(index)
            print(f"✓ 已移除Agent: {getattr(removed, 'name', type(removed).__name__)}")
        else:
            print(f"✗ 无效的索引: {index}")
    
    def get_agents(self) -> List[BaseAgent]:
        """
        获取所有Agent
        
        Returns:
            Agent列表
        """
        return self.agents.copy()
    
    def clear_agents(self) -> None:
        """
        清空所有Agent
        """
        self.agents.clear()
        print("✓ 已清空所有Agent")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Returns:
            执行历史列表
        """
        return self.execution_history.copy()
    
    def clear_history(self) -> None:
        """
        清空执行历史
        """
        self.execution_history.clear()
        print("✓ 已清空执行历史")
    
    def record_execution(self, agent_name: str, task: str, result: Any) -> None:
        """
        记录执行过程
        
        Args:
            agent_name: Agent名称
            task: 任务描述
            result: 执行结果
        """
        record = {
            "agent": agent_name,
            "task": task,
            "result": result,
            "timestamp": None  # TODO: 添加时间戳
        }
        self.execution_history.append(record)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行多Agent协作任务（子类必须实现）
        """
        pass
    
    def print_summary(self) -> None:
        """
        打印执行摘要
        """
        print("\n" + "=" * 60)
        print("多Agent协作摘要")
        print("=" * 60)
        print(f"Agent数量: {len(self.agents)}")
        print(f"执行记录数: {len(self.execution_history)}")
        
        if self.agents:
            print("\nAgent列表:")
            for i, agent in enumerate(self.agents, 1):
                agent_name = getattr(agent, 'name', type(agent).__name__)
                print(f"  {i}. {agent_name}")
        
        print("=" * 60)


class SequentialCoordinator(MultiAgentCoordinator):
    """
    顺序执行协调器
    按顺序执行多个Agent的任务
    """
    
    def execute(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """
        顺序执行任务
        
        Args:
            tasks: 任务列表，每个任务包含agent_index和task_params
            
        Returns:
            结果列表
        """
        print("\n" + "=" * 60)
        print("开始顺序执行多Agent任务")
        print("=" * 60)
        
        results = []
        
        for i, task_config in enumerate(tasks, 1):
            agent_index = task_config.get('agent_index', 0)
            task_params = task_config.get('params', {})
            task_desc = task_config.get('description', f'Task {i}')
            
            if agent_index >= len(self.agents):
                print(f"\n✗ 任务 {i}: 无效的Agent索引 {agent_index}")
                results.append(None)
                continue
            
            agent = self.agents[agent_index]
            agent_name = getattr(agent, 'name', type(agent).__name__)
            
            print(f"\n📌 任务 {i}/{len(tasks)}: {task_desc}")
            print(f"   使用Agent: {agent_name}")
            
            try:
                result = agent.run(**task_params)
                results.append(result)
                self.record_execution(agent_name, task_desc, result)
                print(f"✓ 任务完成")
            except Exception as e:
                print(f"✗ 任务失败: {e}")
                results.append(None)
        
        print("\n" + "=" * 60)
        print("所有任务执行完成")
        print("=" * 60)
        
        return results


class ParallelCoordinator(MultiAgentCoordinator):
    """
    并行执行协调器（框架，待实现）
    """
    
    def execute(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """
        并行执行任务（待实现）
        
        Args:
            tasks: 任务列表
            
        Returns:
            结果列表
        """
        print("⚠ ParallelCoordinator尚未实现")
        print("提示: 可以使用多线程或多进程实现并行执行")
        return []


class PipelineCoordinator(MultiAgentCoordinator):
    """
    流水线协调器（框架，待实现）
    前一个Agent的输出作为后一个Agent的输入
    """
    
    def execute(self, initial_input: Any) -> Any:
        """
        流水线执行（待实现）
        
        Args:
            initial_input: 初始输入
            
        Returns:
            最终输出
        """
        print("⚠ PipelineCoordinator尚未实现")
        print("提示: 可以将前一个Agent的输出传递给下一个Agent")
        return None
