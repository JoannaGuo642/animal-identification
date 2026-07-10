import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QMessageBox,
                               QSlider, QDialog, QVBoxLayout, QLabel, QPushButton)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QBrush,QIcon
from PyQt5.QtCore import QTimer, Qt
from window import Ui_Dialog
import yolov8

class ResultDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("识别结果展示窗口")
        self.resize(850, 620)


        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        self.show_img_label = QLabel()
        self.show_img_label.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel("")
        self.text_label.setStyleSheet("font-size:14px; color:#222;")

        self.btn_close = QPushButton("关闭此窗口")
        self.btn_close.setStyleSheet("background:#555; color:white; padding:8px; border-radius:4px;")
        self.btn_close.clicked.connect(self.close)

        layout.addWidget(self.show_img_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.btn_close)
        self.setLayout(layout)

    # 外部调用刷新图片文字
    def update_content(self, img_path, text):
        pix = QPixmap(img_path)
        self.show_img_label.setPixmap(pix.scaled(800, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.text_label.setText(text)

# ====================== 主窗口类 ======================
class mainWindow(QWidget, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("动物识别检测系统 王晓悦 刘美辛 刘娇如 刘佳 朱美璐")

        self.setWindowIcon(QIcon("img2.jpg"))

        # 背景初始化
        self.bg_pic = QPixmap("img4.jpg")
        self.window_palette = QPalette()
        self.update_bg()

        # 全局变量（
        self.picture_files = []
        self.current_index = 0
        self.cam_timer = QTimer(self)
        self.cap = None
        self.current_detect_img_path = ""
        self.all_detect_result = []
        self.conf_threshold = 0.5
        self.animal_total_count = {}
        self.batch_running = False
        # 批量专用弹窗，全局只创建一次
        self.batch_dialog = ResultDialog(self)
        # 自动切换定时器：3000毫秒=3秒切换一张
        self.auto_switch_timer = QTimer(self)
        self.auto_switch_timer.setInterval(3000)
        self.auto_switch_timer.timeout.connect(self.batch_next_frame)
        # 按钮绑定
        self.pushButton_train.clicked.connect(self.modelTraining)
        self.pushButton_single.clicked.connect(self.singleAnimalDetect)
        self.pushButton_batch.clicked.connect(self.batchSorting)
        self.pushButton_camera.clicked.connect(self.openCamera)
        self.pushButton_export.clicked.connect(self.exportReport)
        self.pushButton_clear.clicked.connect(self.clearAll)
        self.pushButton_saveimg.clicked.connect(self.saveDetectImg)
        # 开始、暂停批量按钮绑定
        self.pushButton_start_batch.clicked.connect(self.start_batch)
        self.pushButton_pause_batch.clicked.connect(self.pause_batch)

        # 滑块绑定
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self.updateConfThreshold)

        # 摄像头定时器
        self.cam_timer.timeout.connect(self.cameraFrameUpdate)

        # 原有按钮样式完全保留
        self.pushButton_train.setStyleSheet("background-color: #f44336; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_single.setStyleSheet("background-color: #2196F3; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_batch.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_camera.setStyleSheet("background-color: #ff9800; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_export.setStyleSheet("background-color: #9c27b0; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_clear.setStyleSheet("background-color: #607d8b; color: white; border: none; border-radius: 5px; padding: 10px;")
        self.pushButton_saveimg.setStyleSheet("background-color: #009688; color: white; border: none; border-radius: 5px; padding: 10px;")

        # 启停按钮样式
        self.pushButton_start_batch.setStyleSheet("background:#00C853;color:white;border:none;border-radius:5px;padding:10px;")
        self.pushButton_pause_batch.setStyleSheet("background:#FF9100;color:white;border:none;border-radius:5px;padding:10px;")

    # 背景自适应更新函数
    def update_bg(self):
        win_w = self.width()
        win_h = self.height()
        scaled_img = self.bg_pic.scaled(win_w, win_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.window_palette.setBrush(QPalette.Window, QBrush(scaled_img))
        self.setPalette(self.window_palette)

    # 拖动窗口大小时自动刷新背景
    def resizeEvent(self, event):
        self.update_bg()
        super().resizeEvent(event)

    # 功能1：更新置信度阈值滑块
    def updateConfThreshold(self, val):
        self.conf_threshold = val / 100
        self.label_slider_val.setText(f"当前阈值：{self.conf_threshold:.2f}")

    # 功能2：动物模型训练（增加异常捕获）
    def modelTraining(self):
        try:
            yolov8.train_your_model(yolov8.DATA_YAML, epochs=3)
            QMessageBox.information(self, "完成", "动物检测模型训练结束！")
        except Exception as e:
            QMessageBox.critical(self, "训练报错", str(e))

    # 功能3：单张图片识别（独立弹窗，不受批量逻辑影响）
    def singleAnimalDetect(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, "选择动物图片", "", "Picture(*.png *.jpg *.jpeg)")
            if not filename:
                QMessageBox.warning(self, "警告", "请选择一张图片！")
                return
            draw_img_path, animal_cnt = yolov8.yolo_detect_draw(filename, self.conf_threshold)
            self.current_detect_img_path = draw_img_path
            res_text = f"图片路径：{filename}\n检测动物：{animal_cnt}"

            self.label_result.setText(res_text)
            self.textEdit_log.append(res_text)
            self.all_detect_result.append(res_text)
            for k, v in animal_cnt.items():
                self.animal_total_count[k] = self.animal_total_count.get(k, 0) + 1
            self.label_count.setText(f"动物统计：{self.animal_total_count}")

            # 单张识别独立弹窗，不共用批量窗口
            single_dialog = ResultDialog(self)
            single_dialog.update_content(draw_img_path, res_text)
            single_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "识别报错", str(e))

    # 功能4：批量文件夹分拣入口（仅加载文件）
    def batchSorting(self):
        try:
            directory = QFileDialog.getExistingDirectory(self, "选择动物图片文件夹")
            if not directory:
                return
            self.picture_files = [
                os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if not self.picture_files:
                QMessageBox.warning(self, "警告", "文件夹无图片！")
                return
            self.current_index = 0
            self.batch_running = False
            self.auto_switch_timer.stop()
            QMessageBox.information(self, "提示", "文件夹加载完成，请点击【开始批量识别】")
        except Exception as e:
            QMessageBox.critical(self, "读取文件夹报错", str(e))

    # 开始批量按钮
    def start_batch(self):
        if len(self.picture_files) == 0:
            QMessageBox.warning(self, "提示", "请先点击【批量文件夹分拣识别】选择图片文件夹！")
            return
        if self.batch_running:
            QMessageBox.information(self, "提示", "批量识别正在运行，无需重复点击")
            return
        self.batch_running = True
        # 先渲染第一张图片到窗口
        self.batch_next_frame()
        # 启动自动切换定时器，窗口打开后自动轮播
        self.auto_switch_timer.start()
        # 显示批量窗口
        self.batch_dialog.show()

    # 暂停批量按钮
    def pause_batch(self):
        if not self.batch_running:
            QMessageBox.warning(self, "提示", "当前没有正在执行的批量任务")
            return
        self.batch_running = False
        self.auto_switch_timer.stop()
        QMessageBox.information(self, "暂停", "批量识别已暂停，点击开始可继续")

    # 定时器触发：刷新单窗口内图片，实现自动播放
    def batch_next_frame(self):
        if not self.batch_running:
            self.auto_switch_timer.stop()
            return
        # 全部图片识别完毕
        if self.current_index >= len(self.picture_files):
            self.batch_running = False
            self.auto_switch_timer.stop()
            self.batch_dialog.close()
            QMessageBox.information(self, "完成", "批量动物分拣全部完毕！")
            return
        try:
            path = self.picture_files[self.current_index]
            draw_img_path, animal_cnt = yolov8.yolo_detect_draw(path, self.conf_threshold)
            self.current_detect_img_path = draw_img_path
            res_text = f"批量图片{self.current_index+1}：{path}\n动物：{animal_cnt}"

            self.label_result.setText(res_text)
            self.textEdit_log.append(res_text)
            self.all_detect_result.append(res_text)
            for k, v in animal_cnt.items():
                self.animal_total_count[k] = self.animal_total_count.get(k, 0) + 1
            self.label_count.setText(f"动物统计：{self.animal_total_count}")

            self.batch_dialog.update_content(draw_img_path, res_text)
            self.current_index += 1
        except Exception as e:
            self.batch_running = False
            self.auto_switch_timer.stop()
            QMessageBox.critical(self, "批量识别报错", str(e))

    # 功能5：摄像头实时识别
    def openCamera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cam_timer.stop()
            self.cap.release()
            self.cap = None
            self.pushButton_camera.setText("打开摄像头实时检测")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头！")
            self.cap = None
            return
        self.pushButton_camera.setText("关闭摄像头")
        self.cam_timer.start(30)

    def cameraFrameUpdate(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = yolov8.camera_detect_frame(frame, self.conf_threshold)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,c = rgb_frame.shape
        q_img = QImage(rgb_frame.data, w, h, w*c, QImage.Format_RGB888)
        self.label_picture.setPixmap(QPixmap.fromImage(q_img).scaled(self.label_picture.size(), Qt.KeepAspectRatio))

    # 功能6：导出批量识别TXT报表
    def exportReport(self):
        try:
            if len(self.all_detect_result) == 0:
                QMessageBox.warning(self, "提示", "暂无识别数据可导出！")
                return
            save_dir = QFileDialog.getExistingDirectory(self, "选择报表保存文件夹")
            if not save_dir:
                return
            save_path = yolov8.export_result_txt(save_dir, self.all_detect_result)
            QMessageBox.information(self, "导出成功", f"报表已保存至：{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出报错", str(e))

    # 功能7：保存带检测框的识别图片
    def saveDetectImg(self):
        try:
            if not self.current_detect_img_path or not os.path.exists(self.current_detect_img_path):
                QMessageBox.warning(self, "提示", "没有识别图片可保存！")
                return
            save_path, _ = QFileDialog.getSaveFileName(self, "保存识别图", "", "图片(*.jpg *.png)")
            if save_path:
                pix = QPixmap(self.current_detect_img_path)
                pix.save(save_path)
                QMessageBox.information(self, "保存成功", "识别图片已保存！")
        except Exception as e:
            QMessageBox.critical(self, "保存报错", str(e))

    # 功能8：一键清空所有数据
    def clearAll(self):
        self.picture_files = []
        self.current_index = 0
        self.all_detect_result = []
        self.animal_total_count = {}
        self.current_detect_img_path = ""
        self.timer.stop()
        self.cam_timer.stop()
        self.auto_switch_timer.stop()
        self.batch_running = False
        self.batch_dialog.close()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.label_picture.clear()
        self.label_result.setText("识别结果：无")
        self.label_count.setText("动物统计：无")
        self.textEdit_log.clear()
        self.pushButton_camera.setText("打开摄像头实时检测")

    # 窗口关闭释放摄像头资源
    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cam_timer.stop()
        self.timer.stop()
        self.auto_switch_timer.stop()
        self.batch_dialog.close()
        event.accept()

# 全局顶层异常捕获，任何错误弹窗提示，不会直接闪退
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = mainWindow()
        win.show()
        sys.exit(app.exec_())
    except Exception as top_err:
        print("全局程序异常：", top_err)