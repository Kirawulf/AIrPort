import os
import cv2
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from ultralytics import YOLO
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from openpyxl import Workbook

# Определяем базовый путь
BASE_DIR = Path(__file__).resolve().parent

# Инициализация Flask с указанием папки шаблонов
app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

# Создаем необходимые папки
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            object_count INTEGER NOT NULL,
            processing_time REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    start_time = datetime.now()
    
    # Получаем файл
    file = request.files['image']
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Проверяем расширение файла
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Генерируем уникальное имя файла
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"uploads/input_{timestamp}.jpg"
    output_filename = f"results/output_{timestamp}.jpg"
    
    # Сохраняем входное изображение
    file.save(input_filename)
    
    try:
        # Обработка изображения
        img = cv2.imread(input_filename)
        results = model(img, classes=24)  # 24 = рюкзак
        
        # Визуализация результатов
        annotated_img = results[0].plot()
        cv2.imwrite(output_filename, annotated_img)
        
        # Подсчет объектов
        count = len(results[0].boxes)
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    # Расчет времени обработки
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Сохраняем запрос в базу данных
    try:
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO requests (timestamp, input_path, output_path, object_count, processing_time) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), input_filename, output_filename, count, processing_time)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    
    return jsonify({
        'count': count,
        'result_image': output_filename,
        'processing_time': processing_time
    })

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory('results', filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory('uploads', filename)

@app.route('/history')
def show_history():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM requests ORDER BY timestamp DESC")
    history = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', history=history)

@app.route('/report/pdf')
def generate_pdf_report():
    try:
        # Получаем данные из БД
        conn = sqlite3.connect('history.db')
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY timestamp DESC", conn)
        conn.close()
        
        # Создаем PDF в памяти
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Заголовок
        elements.append(Paragraph("Отчет по детекции рюкзаков", styles['Title']))
        elements.append(Paragraph(f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        
        # Таблица с данными
        data = [['Дата', 'Входной файл', 'Кол-во объектов', 'Время (сек)']]
        for _, row in df.iterrows():
            data.append([
                row['timestamp'][:19],  # Обрезаем миллисекунды
                os.path.basename(row['input_path']),
                str(row['object_count']),
                f"{row['processing_time']:.2f}"
            ])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(table)
        
        # Статистика
        elements.append(Paragraph(f"<br/><b>Всего запросов:</b> {len(df)}", styles['Normal']))
        elements.append(Paragraph(f"<b>Обнаружено объектов:</b> {df['object_count'].sum()}", styles['Normal']))
        elements.append(Paragraph(f"<b>Среднее время обработки:</b> {df['processing_time'].mean():.2f} сек", styles['Normal']))
        
        # Пример изображения (первая запись)
        if not df.empty:
            img_path = df.iloc[0]['output_path']
            if os.path.exists(img_path):
                elements.append(Paragraph("<br/><b>Пример обнаружения:</b>", styles['Normal']))
                elements.append(Image(img_path, width=400, height=300))
        
        # Генерация PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='backpack_detection_report.pdf',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

@app.route('/report/excel')
def generate_excel_report():
    try:
        # Получаем данные из БД
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM requests ORDER BY timestamp DESC")
        history = cursor.fetchall()
        conn.close()
        
        # Создаем Excel в памяти
        buffer = BytesIO()
        
        # Создаем Excel-книгу
        wb = Workbook()
        ws = wb.active
        ws.title = "История запросов"
        
        # Заголовки
        headers = ["ID", "Дата/Время", "Входной файл", "Выходной файл", "Кол-во объектов", "Время обработки (сек)"]
        ws.append(headers)
        
        # Данные
        for record in history:
            ws.append([
                record[0],
                record[1],
                os.path.basename(record[2]),
                os.path.basename(record[3]),
                record[4],
                record[5]
            ])
        
        # Лист со статистикой
        ws_stats = wb.create_sheet(title="Статистика")
        ws_stats.append(["Метрика", "Значение"])
        ws_stats.append(["Всего запросов", len(history)])
        
        # Рассчитываем общее количество объектов
        total_objects = sum(record[4] for record in history)
        ws_stats.append(["Обнаружено объектов", total_objects])
        
        # Рассчитываем среднее время обработки
        avg_time = sum(record[5] for record in history) / len(history) if history else 0
        ws_stats.append(["Среднее время обработки (сек)", avg_time])
        
        # Сохраняем в буфер
        wb.save(buffer)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='backpack_detection_report.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    except Exception as e:
        return jsonify({"error": f"Excel generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)