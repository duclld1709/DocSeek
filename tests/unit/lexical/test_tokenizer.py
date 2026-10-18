import pytest

from docseek.lexical.tokenizer import Tokenizer, TokenizerConfig, split_identifier

RAW = TokenizerConfig(stopwords=False, stem=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BackgroundTasks", ["backgroundtasks", "background", "tasks"]),
        ("@app.get", ["@app.get", "app", "get"]),
        ("response_model", ["response_model", "response", "model"]),
        ("HTTPException", ["httpexception", "http", "exception"]),
        ("fastapi.Depends()", ["fastapi.depends", "fastapi", "depends"]),
        ("Use FastAPI.", ["use", "fastapi", "fast", "api"]),  # no trailing dot
        ("HTTP 404 on 3.12", ["http", "404", "on", "3.12"]),
    ],
)
def test_code_aware_keeps_whole_identifier_and_parts(text: str, expected: list[str]) -> None:
    assert Tokenizer(RAW).tokenize(text) == expected


def test_plain_mode_splits_on_non_alphanumerics() -> None:
    tok = Tokenizer(TokenizerConfig(stopwords=False, stem=False, code_aware=False))
    assert tok.tokenize("response_model BackgroundTasks @app.get") == [
        "response",
        "model",
        "backgroundtasks",
        "app",
        "get",
    ]


def test_stemming_and_stopwords_apply_to_words_and_parts_only() -> None:
    tok = Tokenizer(TokenizerConfig())
    assert tok.tokenize("What is the path of running dependencies?") == ["path", "run", "depend"]
    # The whole identifier is never stemmed; its parts are.
    assert tok.tokenize("BackgroundTasks") == ["backgroundtasks", "background", "task"]
    assert tok.tokenize("response_models") == ["response_models", "respons", "model"]


def test_case_preserved_when_lowercase_off() -> None:
    tok = Tokenizer(TokenizerConfig(lowercase=False, stopwords=False, stem=False))
    assert tok.tokenize("BackgroundTasks") == ["BackgroundTasks", "Background", "Tasks"]


def test_split_identifier() -> None:
    assert split_identifier("OAuth2PasswordBearer") == ["O", "Auth", "2", "Password", "Bearer"]
    assert split_identifier("get_db_session") == ["get", "db", "session"]
