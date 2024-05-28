import json
import base64
import os
import requests
from typing import AsyncIterable
import fastapi_poe as fp
from modal import Image, Stub, asgi_app, Secret
import logging
import time
from datetime import datetime
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
import io


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def truncate_text(text, max_length=20):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

class EchoBot(fp.PoeBot):
    async def get_response_with_context(
        self, request: fp.QueryRequest, context: fp.RequestContext
    ) -> AsyncIterable[fp.PartialResponse]:
        start_time = time.time()
       
        last_message = request.query[-1].content

        
        logger.info(f"Received request from {context.http_request.client.host}")
        logger.info(f"Method: {context.http_request.method}, URL: {context.http_request.url}")
        logger.info(f"Headers: {context.http_request.headers}")
        logger.info(f"Query Params: {context.http_request.query_params}")
        logger.info(f"Last Message: {last_message}")

        if last_message.startswith("/generate"):
            lines = last_message.split("\n")
            prompt = lines[0][10:].strip()

            negative_prompt = None
            image = None
            strength = None
            model = None
            seed = None
            output_format = None  
            aspect_ratio = None

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Negative Prompt:"):
                    negative_prompt = line.split(":", 1)[1].strip()
                elif line.startswith("Strength:"):
                    strength = float(line.split(":", 1)[1].strip())
                elif line.startswith("Model:"):
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("Seed:"):
                    seed = int(line.split(":", 1)[1].strip())
                elif line.startswith("Output Format:"):
                    output_format = line.split(":", 1)[1].strip()
                elif line.startswith("Aspect Ratio:"):
                    aspect_ratio = line.split(":", 1)[1].strip()

            if request.query[-1].attachments:
                attachment = request.query[-1].attachments[0]
                attachment_url = attachment.url
                response = requests.get(attachment_url)
                if response.status_code == 200:
                    image = response.content
                else:
                    raise Exception(f"Failed to download attachment from URL: {attachment_url}")
                
                
                if strength is None:
                    strength = 0.5
            else:
                image = None

            try:
                api_key = os.environ["STABILITY_API_KEY"]  
                logger.info(f"Generating image for prompt: {prompt}")
                logger.info(f"Negative prompt: {negative_prompt}")
                image_data, filename = self.generate_image(
                    prompt, api_key, negative_prompt, image, strength, model, seed, output_format, aspect_ratio
                )

                await self.post_message_attachment(
                    message_id=request.message_id,
                    file_data=image_data,
                    filename=filename
                )

                response_text = "Image generated and attached."
                yield fp.PartialResponse(text=response_text)
            except Exception as e:
                logger.exception("An exception occurred during image generation:")
                response_text = str(e)
                yield fp.PartialResponse(text=response_text)
                
        elif last_message.startswith("/enhance"):
            input_text = last_message[9:]  
            logger.info(f"Enhancing input text: {input_text}")
            
            prompt = input_text
            query_messages = [fp.ProtocolMessage(role="user", content=prompt)]

            query_request = fp.QueryRequest(
                version="1.1",
                type="query",
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                message_id=request.message_id,
                query=query_messages
            )

            enhanced_prompt = ""
            async for msg in fp.stream_request(
                query_request, "ReversePromptGuide", request.access_key
            ):
                enhanced_prompt += msg.text

            
            start_tag = "<final prompt>"
            end_tag = "</final prompt>"
            start_idx = enhanced_prompt.find(start_tag) + len(start_tag)
            end_idx = enhanced_prompt.find(end_tag)
            final_prompt = enhanced_prompt[start_idx:end_idx].strip()

            logger.info(f"Extracted final prompt: {final_prompt}")

            
            poor_mans_prompt_messages = [fp.ProtocolMessage(role="user", content=final_prompt)]
            poor_mans_prompt_request = fp.QueryRequest(
                version="1.1",
                type="query",
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                message_id=request.message_id,
                query=poor_mans_prompt_messages
            )

            poor_mans_response = ""
            async for msg in fp.stream_request(
                poor_mans_prompt_request, "PoorMansPrompts", request.access_key
            ):
                poor_mans_response += msg.text

            logger.info(f"Response from PoorMansPrompts: {poor_mans_response}")
            yield fp.PartialResponse(text=f"Here is the response from PoorMansPrompts based on your enhanced prompt:\n\n{poor_mans_response}")
        
        elif last_message.startswith("/mojo"):
            user_message = last_message[6:].strip()  
            logger.info(f"Forwarding message to Mojo_Infinity: {user_message}")
            
            mojo_messages = [fp.ProtocolMessage(role="user", content=user_message)]
            mojo_request = fp.QueryRequest(
                version="1.1",
                type="query",
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                message_id=request.message_id,
                query=mojo_messages
            )

            async for msg in fp.stream_request(
                mojo_request, "Mojo_Infinity", request.access_key
            ):
                yield msg
        
        elif last_message.startswith("/fireworks"):
            lines = last_message.split("\n")
            prompt = lines[0][11:].strip()

            
            negative_prompt = None
            height = None
            width = None
            cfg_scale = None
            sampler = None
            samples = None
            seed = None
            steps = None
            safety_check = None
            output_image_format = None

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Negative Prompt:"):
                    negative_prompt = line.split(":", 1)[1].strip()
                elif line.startswith("Height:"):
                    height = int(line.split(":", 1)[1].strip())
                elif line.startswith("Width:"):
                    width = int(line.split(":", 1)[1].strip())
                elif line.startswith("CFG Scale:"):
                    cfg_scale = float(line.split(":", 1)[1].strip())
                elif line.startswith("Sampler:"):
                    sampler = line.split(":", 1)[1].strip()
                elif line.startswith("Samples:"):
                    samples = int(line.split(":", 1)[1].strip())
                elif line.startswith("Seed:"):
                    seed = int(line.split(":", 1)[1].strip())
                elif line.startswith("Steps:"):
                    steps = int(line.split(":", 1)[1].strip())
                elif line.startswith("Safety Check:"):
                    safety_check_value = line.split(":", 1)[1].strip().lower()
                    safety_check = safety_check_value == "true"
                elif line.startswith("Output Image Format:"):
                    output_image_format = line.split(":", 1)[1].strip()
                    if output_image_format not in ["JPEG", "PNG"]:
                        output_image_format = "JPEG"

            try:
                api_key = os.environ["FIREWORKS_API_KEY"]  
                logger.info(f"Generating image using Fireworks AI API for prompt: {prompt}")
                logger.info(f"Negative prompt: {negative_prompt}")
                image_data, filename = await self.generate_fireworks_image(
                    prompt, api_key, negative_prompt, height, width, cfg_scale, sampler, samples, seed, steps, safety_check, output_image_format
                )

                await self.post_message_attachment(
                    message_id=request.message_id,
                    file_data=image_data,
                    filename=filename
                )

                response_text = "Image generated using Fireworks AI API and attached."
                yield fp.PartialResponse(text=response_text)
            except Exception as e:
                logger.exception("An exception occurred during Fireworks AI image generation:")
                response_text = f"Error: {str(e)}"
                yield fp.PartialResponse(text=response_text)
        
        else:
            logger.info("Forwarding request to GPT-3.5-Turbo")
            async for msg in fp.stream_request(
                request, "GPT-3.5-Turbo", request.access_key
            ):
                yield msg
        
        
        resend_text = truncate_text(last_message)
        
        yield fp.PartialResponse(
        text=last_message,
        is_suggested_reply=True,
        data={"display_text": resend_text}
        )

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Request processed in {execution_time:.2f} seconds")
        
    def generate_image(self, prompt, api_key, negative_prompt=None, image=None, strength=None, model=None, seed=None, output_format=None, aspect_ratio=None):
        logger.info(f"Making request to Stability AI API")
        logger.info(f"Positive prompt: {prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Image: {image}")
        logger.info(f"Strength: {strength}")
        logger.info(f"Model: {model}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Output Format: {output_format}")
        logger.info(f"Aspect Ratio: {aspect_ratio}")

        data = {
            "prompt": prompt,
            "model": model,
            "seed": seed,
        }

        if output_format:
            data["output_format"] = output_format

        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        if aspect_ratio:
            data["aspect_ratio"] = aspect_ratio

        files = {"none": ''}

        if image:
            data["mode"] = "image-to-image"
            data["strength"] = strength
            files["image"] = ("image.png", image, "image/png")
        else:
            data["mode"] = "text-to-image"

        logger.info(f"Request data: {data}")

        
        if model == "sd":
            endpoint = "https://api.stability.ai/v2beta/stable-image/generate/core"
        else:
            endpoint = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

        response = requests.post(
            endpoint,
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/*"
            },
            files=files,
            data=data,
        )

        if response.status_code == 200:
            output_format = output_format or "jpeg"  
            current_date = datetime.now().strftime('%Y/%m/%d')  
            timestamp = int(time.time())
            filename = f"{current_date}/generated_image_{timestamp}.{output_format}"
            logger.info(f"Successful response from Stability AI API")
            logger.info(f"Generated image filename: {filename}")

            
            aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            s3 = boto3.client(
                "s3",)
            bucket_name = "echobucket69"  
            try:
                s3.put_object(Body=response.content, Bucket=bucket_name, Key=filename)
                logger.info(f"Image saved to S3: {bucket_name}/{filename}")
            except ClientError as e:
                logger.error(f"Error saving image to S3: {e}")

            return response.content, filename
        else:
            logger.error(f"Error response from Stability AI API: {response.text}")
            raise Exception(response.text)
        
    async def generate_fireworks_image(self, prompt, api_key, negative_prompt=None, height=None, width=None, cfg_scale=None, sampler=None, samples=None, seed=None, steps=None, safety_check=None, output_image_format=None):
        logger.info(f"Making request to Fireworks AI API")
        logger.info(f"Positive prompt: {prompt}")
        logger.info(f"Negative prompt: {negative_prompt}")
        logger.info(f"Height: {height}")
        logger.info(f"Width: {width}")
        logger.info(f"CFG Scale: {cfg_scale}")
        logger.info(f"Sampler: {sampler}")
        logger.info(f"Samples: {samples}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Steps: {steps}")
        logger.info(f"Safety Check: {safety_check}")
        logger.info(f"Output Image Format: {output_image_format}")

        try:
            import fireworks.client
            from fireworks.client.image import ImageInference, Answer

            
            fireworks.client.api_key = api_key
            inference_client = ImageInference(model="stable-diffusion-xl-1024-v1-0")

            if output_image_format is None:
                output_image_format = "PNG"  
            
            
            answer: Answer = await inference_client.text_to_image_async(
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale if cfg_scale is not None else 7.0,  
                height=height if height is not None else 1024,  
                width=width if width is not None else 1024,  
                sampler=sampler,
                steps=steps if steps is not None else 50,  
                seed=seed if seed is not None else 0,  
                safety_check=safety_check if safety_check is not None else True,  
                output_image_format=output_image_format
            )

            if answer.image is None:
                raise RuntimeError(f"No return image, {answer.finish_reason}")
            else:
                current_date = datetime.now().strftime('%Y/%m/%d')  
                timestamp = int(time.time())
                filename = f"{current_date}/fireworks_generated_image_{timestamp}.{output_image_format.lower()}"

                logger.info(f"Successful response from Fireworks AI API")
                logger.info(f"Generated image filename: {filename}")

                
                aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
                aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
                s3 = boto3.client(
                    "s3",
                )
                bucket_name = "echobucket69"  
                try:
                    
                    image_bytes = io.BytesIO()
                    answer.image.save(image_bytes, format=output_image_format)
                    image_bytes.seek(0)

                    s3.put_object(Body=image_bytes, Bucket=bucket_name, Key=filename)
                    logger.info(f"Image saved to S3: {bucket_name}/{filename}")
                except ClientError as e:
                    logger.error(f"Error saving image to S3: {e}")

                return image_bytes.getvalue(), filename

        except Exception as e:
            logger.error(f"Error generating image with Fireworks AI API: {e}")
            raise
        
    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(
            server_bot_dependencies={"GPT-3.5-Turbo": 1, "ReversePromptGuide": 1, "PoorMansPrompts": 1, "Mojo_Infinity": 1},
            allow_attachments=True,
            introduction_message="""# EchoBot Documentation

EchoBot is a versatile bot that allows users to generate images using the Stability AI API, enhance prompts using ReversePromptGuide and PoorMansPrompts, and interact with the Mojo_Infinity bot.

## Functionalities

### 1. Image Generation (`/generate`)

To generate an image, use the `/generate` command followed by your desired prompt. You can also specify optional parameters to control the image generation process. Details of the available parameters are at the end of this message.

Example:
```javascript
/generate A beautiful sunset over the ocean
Negative Prompt: low quality, blurry
Model: sd3
```

**Important quirks to keep in mind:**
- The parameters are case-sensitive. Make sure to use the correct capitalization for each parameter.
- The parameter values should be on the same line as the parameter name, without any new lines or paragraphs in between. For example, "Negative Prompt: low quality, blurry" should be on a single line.
- If you don't specify a parameter, it will revert to its default value.

### 2. Prompt Enhancement (`/enhance`)

To enhance a prompt, use the `/enhance` command followed by the prompt you want to enhance.

Example:
```javascript
/enhance A mysterious castle in a misty forest
```

The bot will generate an enhanced prompt using ReversePromptGuide and then send it to PoorMansPrompts to generate a weighted and keyword-focused response based on the iterated enhanced prompt. This only outputs an enhanced prompt. To generate an image, use `/generate` with the enhanced prompt.

### 3. Mojo_Infinity Interaction (`/mojo`)

To generate an image with the Mojo_Infinity bot, use the `/mojo` command followed by your message.

Example:
```javascript
/mojo a dog wearing a hat
```

---

## Usage

To ensure correct parsing of your commands and parameters, it's recommended to copy-paste the provided examples and modify them according to your needs.

Remember:
- Use `/generate` to generate images
- Use `/enhance` to enhance prompts
- Use `/mojo` to generate images with Mojo_Infinity

I fund this bot with my own money, so please be considerate with your usage &/or subscribe if you can. Thank you! ❤️

## Update 5/3/2024

[X] Image-to-image is fixed and supports all of the same parameters as text-to-image now.

For clarity, the use of all parameters available looks like this:

```javascript
/generate an absolute chad, sitting at a computer, strong jawline, confident expression
Negative Prompt: cartoon, anime, sketch, low quality
Strength: 0.5
Model: sd3
Seed: 0
Output Format: png
Aspect Ratio: 1:1
```

Everything but `/generate` is optional *unless* you're doing image-to-image, in which case you need to specify a Strength parameter.

**Parameter Options:**
- Negative Prompt: `text` *defaults to None*
- Strength: `number between 0 and 1` *defaults to 0.5*
- Aspect Ratio: `16:9`, `1:1`, `21:9`, `2:3`, `3:2`, `4:5`, `5:4`, `9:16`, `9:21` *default to 1:1*
- Output Format: `jpeg`, `png` *I think Poe defaults to png*
- Model: `sd3`, `sd3-turbo` *defaults to sd3*
- Seed: `number` *defaults to 0, which means random seed*
"""
                            
        )

REQUIREMENTS = ["fastapi-poe==0.0.36", "requests", "boto3", "fireworks-ai"]
image = Image.debian_slim().pip_install(*REQUIREMENTS)
stub = Stub("echobot-poe")

@stub.function(image=image, secrets=[Secret.from_name("echobot-secrets")])
@asgi_app()
def fastapi_app():
    POE_ACCESS_KEY = os.environ["POE_ACCESS_KEY"]  
    bot = EchoBot(access_key=POE_ACCESS_KEY)
    app = fp.make_app(bot)
    return app