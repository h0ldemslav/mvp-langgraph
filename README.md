# mvp-langgraph

Instructions for running the generated MVP and information about its architecture can be found in [docs/README.md](./docs/README.md).

## Architecture

The MVP is implemented using [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview). Below is rough sketch of the agent flow:

```
START
  |
  +-- M2 --> architect --> [HUMAN REVIEW] -- accept --> END
  |                           |   ^
  |                           |   +-- feedback --> architect
  |
  +-- M3.x --> backend_dev --> frontend_dev --> qa_reviewer
                   ^                ^                |
                   | QA retry       | QA retry       +-- tests pass or iteration limit
                   | backend_dev    | frontend_dev                   |
                   +----------------+                                v
                                                              [HUMAN REVIEW]
                                                                |        |
                                                           accept        +-- feedback
                                                                |              |
                                                                v              +-- M3.x --> selected dev
                                                               END
```

In M2, a person approves the architect's output; in M3.x, QA reviews the result. Human feedback sends the output back to the architect in M2, while in M3.x, a person directs it to the backend or frontend.
