# server.py
import os
import json
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, JSONData
from agents import research, synth, memory_store, memory, vectorstore
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Global MCP object required by mcp dev/run
mcp = FastMCP()

# Initialize OTEL (sends to localhost:4317 by default)
def init_otel():
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    provider = TracerProvider()
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)

init_otel()

@mcp.action(name="query_second_brain", description="Run a multi-agent query pipeline")
def query_second_brain(payload: JSONData) -> JSONData:
    """
    payload: {"user": "name", "prompt": "your question"}
    returns: {"answer": str, "research": str, "retrieved": list}
    """
    user = payload.get("user", "me")
    prompt = payload.get("prompt", "")
    # 1) Research
    findings = research(prompt)
    # 2) Retrieve memory/context via RAG
    retrieved = vectorstore.search(prompt, k=3)
    # 3) Synthesize
    answer = synth(prompt, findings, retrieved)
    # 4) Store memory (scrubbed internally)
    memory_store(user, prompt, answer)
    return {"answer": answer, "research": findings, "retrieved": retrieved}

# Simple health action
@mcp.action(name="health", description="Health check")
def health(_: dict) -> TextContent:
    return TextContent("OK")
