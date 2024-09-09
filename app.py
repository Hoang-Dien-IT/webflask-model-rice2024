from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Tải model và scaler đã lưu
model = joblib.load('random_forest_model_rice.pkl')
scaler = joblib.load('scaler_rice.pkl')  # scaler đã được huấn luyện trên tập dữ liệu gốc

@app.route('/')
def home():
    # Khởi tạo biến class_0_prob và class_1_prob là None khi trang chủ được truy cập
    return render_template('index.html', class_0_prob=None, class_1_prob=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        major_axis_length = float(request.form['major_axis_length'])
        minor_axis_length = float(request.form['minor_axis_length'])
        eccentricity = float(request.form['eccentricity'])
        convex_area = float(request.form['convex_area'])
        extent = float(request.form['extent'])

        # Tạo mảng numpy từ dữ liệu đầu vào
        features = np.array([[area, perimeter, major_axis_length, minor_axis_length, eccentricity, convex_area, extent]])

        # Áp dụng MinMaxScaler
        features_scaled = scaler.transform(features)

        # Dự đoán xác suất của từng lớp nhãn
        probabilities = model.predict_proba(features_scaled)[0]
        class_0_prob = probabilities[0] * 100
        class_1_prob = probabilities[1] * 100

        # Trả kết quả về template
        return render_template('index.html', class_0_prob=class_0_prob, class_1_prob=class_1_prob)

    except Exception as e:
        # Nếu có lỗi xảy ra, không truyền giá trị nào
        return render_template('index.html', class_0_prob=None, class_1_prob=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
