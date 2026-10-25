import base64
json_str = '{"a":"b"}'
print(base64.b64encode(json_str.encode()).decode())
