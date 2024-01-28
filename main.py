from flask import Flask, request, render_template, make_response, jsonify, redirect, url_for
from flask_restful import Resource, Api
import cv2
import numpy as np
import requests
import os

PeopleCounterApp = Flask(__name__)
api = Api(PeopleCounterApp)

# initialize the HOG descriptor/person detector
# src https://thedatafrog.com/en/articles/human-detection-video/
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def process_image(frame):
    frame_name = str(id(frame))  # Unique identifier

    # Detect people and get boxes and weights
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Draw rectangles on the input frame
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the frame with boxes to a file
    output_path = os.path.join('data', f'{frame_name}_boxes.jpg')
    cv2.imwrite(output_path, frame)

    # Return the count and link to the saved image
    response_data = {'count': len(boxes), 'image_link': output_path}
    return response_data


# Example link to try the class
# http://127.0.0.1:5000/people_counter_local
class PeopleCounterLocal(Resource):
    def get(self):
        img = cv2.imread('data/shop.jpg')
        return process_image(img)


# Example link to try the class
# http://127.0.0.1:5000/people_counter_get?url=https://transinfo.pl/wp-content/uploads/2022/01/Zabytkowy-dworzec-PKP-w-Skawinie-po-remoncie-dostepny-dla-podroznych-0-5-851x550.jpg
class PeopleCounterGet(Resource):
    def get(self):
        image_url = request.args.get('url')

        if image_url is None:
            return {'error': 'Please provide an image URL.'}, 400

        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return {'error': f'Failed to download or decode the image. {str(e)}'}, 500

        return process_image(img)


# To use this class use the class below...
class PeopleCounterPost(Resource):
    def post(self):
            # Get the image file from the POST request
            file = request.files['image']
            # Read the image file
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            # Process the image
            result = process_image(img)

            # Check if 'count' and 'image_link' keys are present in the result dictionary
            if 'count' in result and 'image_link' in result:
                # Explicitly jsonify the result data
                result_json = jsonify(result)

                # Render results.html directly with the result data
                headers = {'Content-Type': 'text/html'}
                # return make_response(render_template('results.html', result=result), 200, headers)
                return redirect(url_for('show_results', count=result['count'], image_link=result['image_link']))
                # return result
            else:
                return {'error': 'Invalid result format. Missing count or image_link.'}, 500


# Example link to try the class
# http://127.0.0.1:5000/people_counter_post_front
class PeopleCounterPostFront(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('upload.html'), 200, headers)


# Example route to display results on results.html
@PeopleCounterApp.route('/show_results')
def show_results():
    count = request.args.get('count')
    image_link = request.args.get('image_link')
    headers = {'Content-Type': 'text/html'}
    return make_response(render_template('results.html', count=count, image_link=image_link), 200, headers)


api.add_resource(PeopleCounterLocal, '/people_counter_local')
api.add_resource(PeopleCounterGet, '/people_counter_get')
api.add_resource(PeopleCounterPost, '/people_counter_post')
api.add_resource(PeopleCounterPostFront, '/people_counter_post_front')

if __name__ == '__main__':
    PeopleCounterApp.run(debug=True)
