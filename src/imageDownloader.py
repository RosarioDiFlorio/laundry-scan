from google_images_download import google_images_download  # importing the library

response = google_images_download.googleimagesdownload()  # class instantiation

arguments = {
    "Records": [
        {"keywords":
             "picture of care label",
         "limit": 100,
         "output_directory": "../data/labels",
         "print_urls": False},
        {"keywords":
             "laundry care tag",
         "limit": 100,
         "output_directory": "../data/labels",
         "print_urls": False}
    ]
}  # creating list of arguments

for arg in arguments["Records"]:
    paths = response.download(arg)  # passing the arguments to the function
    print(paths)
