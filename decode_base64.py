import base64

def decode_base64_to_image(base64_string, output_file_path):
    try:
        image_data = base64.b64decode(base64_string)

        with open(output_file_path, "wb") as image_file:
            image_file.write(image_data)

        print(f"Image successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to decode Base64 string: {e}")


base64_string = ""
# Specify output file path and extension (e.g., .png, .jpg)
output_file = "output_image.png"
decode_base64_to_image(base64_string, output_file)
