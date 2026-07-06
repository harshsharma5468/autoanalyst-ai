"""
AutoAnalyst module — multi-format ingestion, EDA, NL query, and SQL generation.
"""
from .data_ingestion import FileProcessor
from .analysis_engine import AnalysisEngine
from .sql_generator import SQLGenerator
from .nl_query import NLQueryEngine

__all__ = ["FileProcessor", "AnalysisEngine", "SQLGenerator", "NLQueryEngine"]
