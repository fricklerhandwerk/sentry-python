import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

import sentry_sdk
from sentry_sdk.consts import SPANDATA
from sentry_sdk.integrations.boto3 import Boto3Integration


@pytest.fixture
def converse_response():
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 50, "outputTokens": 30, "totalTokens": 80},
        "metrics": {"latencyMs": 500},
    }


@pytest.mark.parametrize("span_streaming", [True, False])
def test_converse(sentry_init, capture_items, span_streaming, converse_response):
    sentry_init(
        traces_sample_rate=1.0,
        integrations=[Boto3Integration()],
        trace_lifecycle="stream" if span_streaming else "static",
    )

    session = boto3.Session(
        aws_access_key_id="-", aws_secret_access_key="-", region_name="us-east-1"
    )
    client = session.client("bedrock-runtime")
    stubber = Stubber(client)
    stubber.add_response("converse", converse_response)

    items = capture_items()

    with stubber, sentry_sdk.start_transaction(name="parent"):
        client.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

    sentry_sdk.flush()
    (span,) = (item.payload for item in items if item.type == "span")

    assert span["attributes"][SPANDATA.GEN_AI_PROVIDER_NAME] == "aws.bedrock"
    assert span["attributes"][SPANDATA.GEN_AI_RESPONSE_STREAMING] is False


@pytest.mark.parametrize("span_streaming", [True, False])
def test_converse_error(sentry_init, capture_items, span_streaming):
    sentry_init(
        traces_sample_rate=1.0,
        integrations=[Boto3Integration()],
        trace_lifecycle="stream" if span_streaming else "static",
    )

    session = boto3.Session(
        aws_access_key_id="-", aws_secret_access_key="-", region_name="us-east-1"
    )
    client = session.client("bedrock-runtime")
    original_exception = ClientError(
        error_response={
            "Error": {"Code": "AccessDeniedException", "Message": "Access denied"},
            "ResponseMetadata": {
                "RequestId": "test-id",
                "HostId": "",
                "HTTPStatusCode": 403,
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        },
        operation_name="Converse",
    )

    def raise_client_error(**kwargs):
        raise original_exception

    client.meta.events.register("before-parameter-build", raise_client_error)

    items = capture_items()

    with sentry_sdk.start_transaction(name="parent"), pytest.raises(ClientError):
        client.converse(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

    sentry_sdk.flush()
    (span,) = (item.payload for item in items if item.type == "span")

    assert span["attributes"][SPANDATA.GEN_AI_PROVIDER_NAME] == "aws.bedrock"
    assert span["attributes"][SPANDATA.GEN_AI_OPERATION_NAME] == "chat"

    assert span["attributes"][SPANDATA.ERROR_TYPE] == "AccessDeniedException"
    assert span["attributes"][SPANDATA.HTTP_STATUS_CODE] == 403
    assert span["attributes"][SPANDATA.AWS_REQUEST_ID] == "test-id"
    assert span["status"] == "error"


@pytest.mark.parametrize("span_streaming", [True, False])
def test_converse_stream(sentry_init, capture_items, span_streaming):
    sentry_init(
        traces_sample_rate=1.0,
        integrations=[Boto3Integration()],
        trace_lifecycle="stream" if span_streaming else "static",
    )

    session = boto3.Session(
        aws_access_key_id="-", aws_secret_access_key="-", region_name="us-east-1"
    )
    client = session.client("bedrock-runtime")
    stubber = Stubber(client)
    stubber.add_response("converse_stream", {})

    items = capture_items()

    with stubber, sentry_sdk.start_transaction(name="parent"):
        client.converse_stream(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

    sentry_sdk.flush()
    (span,) = (item.payload for item in items if item.type == "span")

    assert span["attributes"][SPANDATA.GEN_AI_PROVIDER_NAME] == "aws.bedrock"
    assert span["attributes"][SPANDATA.GEN_AI_RESPONSE_STREAMING] is True


@pytest.mark.parametrize("span_streaming", [True, False])
def test_converse_stream_error(sentry_init, capture_items, span_streaming):
    sentry_init(
        traces_sample_rate=1.0,
        integrations=[Boto3Integration()],
        trace_lifecycle="stream" if span_streaming else "static",
    )

    session = boto3.Session(
        aws_access_key_id="-", aws_secret_access_key="-", region_name="us-east-1"
    )
    client = session.client("bedrock-runtime")
    original_exception = ClientError(
        error_response={
            "Error": {"Code": "AccessDeniedException", "Message": "Access denied"},
            "ResponseMetadata": {
                "RequestId": "test-id",
                "HostId": "",
                "HTTPStatusCode": 403,
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        },
        operation_name="ConverseStream",
    )

    def raise_client_error(**kwargs):
        raise original_exception

    client.meta.events.register("before-parameter-build", raise_client_error)

    items = capture_items()

    with sentry_sdk.start_transaction(name="parent"), pytest.raises(ClientError):
        client.converse_stream(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )

    sentry_sdk.flush()
    (span,) = (item.payload for item in items if item.type == "span")

    assert span["attributes"][SPANDATA.GEN_AI_PROVIDER_NAME] == "aws.bedrock"
    assert span["attributes"][SPANDATA.GEN_AI_RESPONSE_STREAMING] is True

    assert span["attributes"][SPANDATA.ERROR_TYPE] == "AccessDeniedException"
    assert span["attributes"][SPANDATA.HTTP_STATUS_CODE] == 403
    assert span["attributes"][SPANDATA.AWS_REQUEST_ID] == "test-id"
    assert span["status"] == "error"
