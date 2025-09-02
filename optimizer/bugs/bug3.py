ERROR:adam_email_routing.utils.bedrock:Could not complete prompt due to: Class must be a subclass of pydantic.BaseModel
ERROR:OfflineCompatibleModelClient:Error calling Bedrock: Could not complete prompt due to: Class must be a subclass of pydantic.BaseModel
Traceback (most recent call last):
  File "/home/jovyan/adam-email-routing/src/main/python/adam_email_routing/utils/bedrock.py", line 39, in create
    resp = client.chat.completions.create(
  File "/home/jovyan/.conda/envs/adalflow/lib/python3.10/site-packages/instructor/client.py", line 366, in create
    return self.create_fn(
  File "/home/jovyan/.conda/envs/adalflow/lib/python3.10/site-packages/instructor/patch.py", line 238, in new_create_sync
    response_model, new_kwargs = handle_response_model(
  File "/home/jovyan/.conda/envs/adalflow/lib/python3.10/site-packages/instructor/process_response.py", line 1222, in handle_response_model
    response_model = prepare_response_model(response_model)
  File "/home/jovyan/.conda/envs/adalflow/lib/python3.10/site-packages/instructor/process_response.py", line 1114, in prepare_response_model
    response_model = openai_schema(response_model)  # type: ignore
  File "/home/jovyan/.conda/envs/adalflow/lib/python3.10/site-packages/instructor/function_calls.py", line 680, in openai_schema
    raise TypeError("Class must be a subclass of pydantic.BaseModel")
TypeError: Class must be a subclass of pydantic.BaseModel

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/jovyan/adam-email-routing/prompt-optimizer/optimizer/model_client.py", line 134, in call
    response = self.bedrock_client.create(**api_kwargs)
  File "/home/jovyan/adam-email-routing/src/main/python/adam_email_routing/utils/bedrock.py", line 54, in create
    raise RuntimeError(f"Could not complete prompt due to: {str(e)}")
RuntimeError: Could not complete prompt due to: Class must be a subclass of pydantic.BaseModel
📊 Classification result: GeneratorOutput(id=None, input=None, data=None, thinking=None, tool_use=None, images=None, error='Could not complete prompt due to: Class must be a subclass of pydantic.BaseModel', usage=None, raw_response=None, api_response=None, metadata=None)
