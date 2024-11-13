from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Tải model và scaler đã lưu
model = joblib.load('random_forest_model_rice.pkl')
df_cleaned = pd.read_excel("Rice2024_cleaned.xlsx", engine='openpyxl')


@app.route('/')
def home():
    # Khởi tạo biến class_0_prob và class_1_prob là None khi trang chủ được truy cập
    return render_template('index.html', class_0_prob=None, class_1_prob=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        area = float(request.form['area'])
        major_axis_length = float(request.form['major_axis_length'])
        minor_axis_length = float(request.form['minor_axis_length'])
        eccentricity = float(request.form['eccentricity'])
        extent = float(request.form['extent'])

        Area = [df_cleaned['Area'].min(), df_cleaned['Area'].max()]
        Major_Axis_Length = [df_cleaned['Major_Axis_Length'].min(), df_cleaned['Major_Axis_Length'].max()]
        Minor_Axis_Length = [df_cleaned['Minor_Axis_Length'].min(), df_cleaned['Minor_Axis_Length'].max()]
        Eccentricity = [df_cleaned['Eccentricity'].min(), df_cleaned['Eccentricity'].max()]
        Extent = [df_cleaned['Extent'].min(), df_cleaned['Extent'].max()]

        area = (area - Area[0]) / (Area[1] - Area[0])
        major_axis_length = (major_axis_length - Major_Axis_Length[0]) / (Major_Axis_Length[1] - Major_Axis_Length[0])
        minor_axis_length = (minor_axis_length - Minor_Axis_Length[0]) / (Minor_Axis_Length[1] - Minor_Axis_Length[0])
        eccentricity = (eccentricity - Eccentricity[0]) / (Eccentricity[1] - Eccentricity[0])
        extent = (extent - Extent[0]) / (Extent[1] - Extent[0])


        features = np.array([[area, major_axis_length, minor_axis_length, eccentricity, extent]])
        print(features)

        probabilities = model.predict_proba(features)[0]
        class_0_prob = probabilities[0] * 100
        class_1_prob = probabilities[1] * 100

        # Trả kết quả về template
        return render_template('index.html', class_0_prob=class_0_prob, class_1_prob=class_1_prob)

    except Exception as e:
        # Nếu có lỗi xảy ra, không truyền giá trị nào
        return render_template('index.html', class_0_prob=None, class_1_prob=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
