import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)

# Allow CORS for all routes
CORS(app, resources={r"/*": {"origins": "https://farzibustersai.netlify.app"}})

@app.route('/check-currency', methods=['POST'])
def check_currency():
    # your logic here
    return jsonify(result="Success")


@app.route('/check-currency', methods=['POST'])
def check_currency():
    try:
        # Get images from the form
        real_image = request.files.get('real_image')
        sample_image = request.files.get('sample_image')

        # Check if both images are uploaded
        if not real_image or not sample_image:
            return jsonify({"message": "Both images must be uploaded."}), 400

        # Read the images into OpenCV
        real_img = cv2.imdecode(np.frombuffer(real_image.read(), np.uint8), cv2.IMREAD_COLOR)
        sample_img = cv2.imdecode(np.frombuffer(sample_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if images were loaded correctly
        if real_img is None or sample_img is None:
            return jsonify({"message": "Error loading one or both images. Please check the uploaded files."}), 400

        # Convert images to grayscale
        gray_real = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        gray_sample = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

        # Use ORB to detect keypoints and descriptors
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_real, None)
        kp2, des2 = orb.detectAndCompute(gray_sample, None)

        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Apply Lowe's Ratio Test to filter out weak matches
        matches = bf.knnMatch(des1, des2, k=2)  # Find the 2 closest matches for each descriptor
        good_matches = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)

        # Check if there are enough good matches (lower the threshold to 5)
        if len(good_matches) > 5:  # Minimum 5 good matches needed
            # Get the keypoints from the good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # RANSAC to estimate homography

            # Use the homography matrix to warp the sample image to the real image perspective
            h, w = gray_real.shape
            warped_sample_img = cv2.warpPerspective(sample_img, M, (w, h))

            # You can optionally display the warped image to see the alignment (for debugging)
            # cv2.imshow("Warped Sample Image", warped_sample_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Now, you can compare the real and warped sample images (you can use different methods)
            # Here, we will do a simple mean squared error or correlation coefficient to measure similarity
            diff = cv2.absdiff(real_img, warped_sample_img)
            result = np.sum(diff)  # Summing the difference as a very basic similarity measure

            if result < 1000000:  # If the difference is small enough, we consider it legitimate
                result_message = "Currency is legitimate"
            else:
                result_message = "Currency is fake"
        else:
            result_message = "Not enough good matches. Unable to detect."

        return jsonify({"message": result_message})

    except Exception as e:
        # Log the error details for debugging
        error_message = traceback.format_exc()
        print(f"Error: {error_message}")  # This logs detailed error info to the console
        return jsonify({"message": "An error occurred during processing.", "error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)





