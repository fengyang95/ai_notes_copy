from notes._exceptions import LLMAPIServiceError


def query_gpt(prompt):
    import requests
    import json
    url = "http://chatverse.bytedance.net/v1/chat"
    headers = requests.get('https://cloud.bytedance.net/auth/api/v1/jwt',
                           headers={'Authorization': 'Bearer 22bd9ec2768c762fd12f605c1965d239'}).headers
    payload = json.dumps({
        # "model": "gpt-35-turbo",
        "model": "gpt-4-0613",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    max_retries = 100
    for attempt in range(max_retries):
        # print(f"Retry {attempt + 1} of {max_retries}")
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()  # 这将触发状态码为4xx和5xx的异常
            data = json.loads(response.text)
            return data["data"]["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as http_err:
            raise LLMAPIServiceError(http_err, prompt)
        except Exception as err:
            raise LLMAPIServiceError(err, prompt)
    return None


if __name__ == '__main__':
    # test
    final_prompt = """hello!你好毛泽东"""
    print(query_gpt(final_prompt))
