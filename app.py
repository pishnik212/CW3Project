import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import *
import os
from werkzeug.utils import secure_filename
from ModelSettings import *
from InputCheck import *
from OutputData import *

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'


def usernameCheck(inputUsername):
    if inputUsername == '':
        resultString = 'Пустое значение пользовательского id!'
        result = False
    elif len(inputUsername) < 8:
        resultString = 'Слишком короткий пользовательский id!'
        result = False
    elif len(inputUsername) > 10:
        resultString = 'Слишком длинный пользовательский id!'
        result = False
    elif not inputUsername.isdigit():
        resultString = 'Пользовательский id должен содержать только цифры!'
        result = False
    else:
        result = True
        resultString = 'OK'
    return result, resultString

def fileExtension(file_path):
    import os
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        return True
    else:
        return False

def fileFormatCheck(f1, f2):
    if not f1.filename.endswith('.csv') or not f2.filename.endswith('.csv'):
        return False
    else:
        return True


@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        inputUsername = request.form['inputUsername']
        userRes, userString = usernameCheck(inputUsername)[0], usernameCheck(inputUsername)[1]
        if not userRes:
            return userString
        session['username'] = int(request.form['inputUsername'])

        f1 = request.files.get('file1')
        f2 = request.files.get('file2')

        if f1 == '':
            return 'Пустой файл'
        if not fileFormatCheck(f1, f2):
            return 'Неверный формат файла! Необходимое расширение *.csv'


        data_filename1 = secure_filename(f1.filename)
        data_filename2 = secure_filename(f2.filename)

        f1.save(os.path.join(app.config['UPLOAD_FOLDER'],
                             data_filename1))
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'],
                             data_filename2))

        session['uploaded_data_file_path_1'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename1)
        session['uploaded_data_file_path_2'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename2)

        path_routes = session.get('uploaded_data_file_path_1', None)
        path_safezones = session.get('uploaded_data_file_path_2', None)
        dataRoutes = pd.read_csv(path_routes,
                                 encoding='unicode_escape')
        dataSafeZones = pd.read_csv(path_safezones,
                                    encoding='unicode_escape')
        # dataRoutes = pd.read_csv('google/routes2_.csv',
        #                          encoding='unicode_escape')
        # dataSafeZones = pd.read_csv('google/safezones2.csv',
        #                             encoding='unicode_escape')

        # inputUsername = int(92914108)
        # session['username'] = inputUsername

        if routes_columns(dataRoutes) and sz_columns(dataSafeZones):
            return render_template('index2.html')
        else:
            return 'Неверная структура файла!'

    return render_template("index.html")


@app.route('/download_txt')
def download_file():
    path = "staticFiles/output_data.txt"
    return send_file(path, as_attachment=True)


@app.route('/download_json')
def download_file1():
    path = "staticFiles/output_data.json"
    return send_file(path, as_attachment=True)

@app.route('/download_csv')
def download_file2():
    path = "staticFiles/output_data.csv"
    return send_file(path, as_attachment=True)

@app.route('/predict')
def predict():
    # dataRoutes = pd.read_csv('google/routes2_.csv', encoding='utf-8')
    # dataSafeZones = pd.read_csv('google/safezones2.csv', encoding='utf-8')
    username = session['username']

    path_safezones = session.get('uploaded_data_file_path_2', None)
    dataSafeZones = pd.read_csv(path_safezones,
                                encoding='unicode_escape')

    path_routes = session.get('uploaded_data_file_path_1', None)
    dataRoutes = pd.read_csv(path_routes,
                             encoding='unicode_escape')

    # dataRoutes = pd.read_csv('google/routes2_.csv', encoding='utf-8')
    sz = user_tables(dataRoutes, username)

    real_points, pred_points, dists, output, xtest = last_function(username, dataRoutes, dataSafeZones)



    real_points, pred_points, dists = real_points[-1], pred_points[-1], dists[-1]
    start_times, train_times, last_times = det_time_arr(dataRoutes, username)

    set_txt(username, real_points[0], real_points[1])
    username, lat, lon = get_txt()

    set_json(username, real_points[0], real_points[1])
    set_csv(username, real_points[0], real_points[1])


    return render_template('predict_result.html',
                           real_points=real_points,
                           pred_points=pred_points,
                           dists=dists,
                           output=output,
                           start_times=start_times,
                           train_times=train_times,
                           last_times=last_times,
                           xtest=xtest,
                           lat = lat,
                           lon = lon,
                           # pred_array=pred_array,
                           # real_array = real_array
                           )

@app.route('/test')
def predict_test():
    # dataRoutes = pd.read_csv('google/routes2_.csv', encoding='utf-8')
    # dataSafeZones = pd.read_csv('google/safezones2.csv', encoding='utf-8')
    username = session['username']

    path_safezones = session.get('uploaded_data_file_path_2', None)
    dataSafeZones = pd.read_csv(path_safezones,
                                encoding='unicode_escape')

    path_routes = session.get('uploaded_data_file_path_1', None)
    dataRoutes = pd.read_csv(path_routes,
                             encoding='unicode_escape')

    # dataRoutes = pd.read_csv('google/routes2_.csv', encoding='utf-8')
    sz = user_tables(dataRoutes, username)

    real_points, pred_points, dists, output, xtest = last_function(username, dataRoutes, dataSafeZones)



    real_points, pred_points, dists = real_points[-1], pred_points[-1], dists[-1]
    start_times, train_times, last_times = det_time_arr(dataRoutes, username)

    set_txt(username, real_points[0], real_points[1])
    username, lat, lon = get_txt()

    set_json(username, real_points[0], real_points[1])
    set_csv(username, real_points[0], real_points[1])


    return render_template('predict_result_test.html',
                           real_points=real_points,
                           pred_points=pred_points,
                           dists=dists,
                           output=output,
                           start_times=start_times,
                           train_times=train_times,
                           last_times=last_times,
                           xtest=xtest,
                           lat = lat,
                           lon = lon,
                           # pred_array=pred_array,
                           # real_array = real_array
                           )


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
