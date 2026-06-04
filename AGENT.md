We are developing code for a research paper.

# Overview
1. Motivation: GPU Idle time during tool calls for batch 1 edge compute LLM agentic workloads is not utilized at all. We want to use this time for computationally intense tasks to improve peformance of the agent.
2. Modern LLM agents have a very limited context, highly dependent on the size of the KV Cache. Methods such as SnapKV, StreamingLLM, etc. are token eviction algorithms which drop (or ``forget'') unimportant tokens from the KV Cache based on attention scores. However, as a result, important details are sometimes forgotten, see arxiv paper: The Pitfalls of KV Cache Compression (https://arxiv.org/abs/2510.00231). 
3. We have ideas for KV Cache repair algorithms, which maintain a list of key-value pairs for evicted tokens (different methods for determing which tokens to store even in this separate list, e.g. based on L2 norms of query vectors), and then recompute attention scores against the most recent tokens' query vectors to determine if evicted tokens are now very relevant. Long tool calls, which are becoming more and more prevalent in agentic workloads (e.g. script calls, MCP tool calls, etc., which often take 2+ seconds), provide an opportunity to perform these KV Cache recomputations without adding to the latency of the agent's response.


# Implementation Rules
1. When implementing code, keep non boilerplate code very segmented and modular, with clear function definitions and a clear logical flow, so that we can easily understand, test,  modify, and debug the code. Be very clear about where code is found, and try to keep it minimal when possible.