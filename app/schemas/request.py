from pydantic import BaseModel, ConfigDict, Field


class SQLGenerationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=3, max_length=1000)
    database_schema: str = Field(alias="schema", min_length=10, max_length=16000)

    @property
    def schema(self):
        return self.database_schema


class SQLGenerationResponse(BaseModel):
    question: str
    database_schema: str = Field(alias="schema")
    sql: str
    latency_ms: int
    model: str
    mode: str
