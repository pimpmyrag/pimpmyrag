"""Compatibility shim: expose SpanMultiTaskModel under the module name
`multitask_model` to match imports used in train_multi_task.py.
"""
from multi_task_model import SpanMultiTaskModel

__all__ = ["SpanMultiTaskModel"]

