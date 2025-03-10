import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                            QHBoxLayout, QTabWidget, QLabel, QComboBox, QSizePolicy,
                            QRadioButton, QLineEdit, QMessageBox, QScrollArea, QPushButton)
from PyQt5.QtCore import Qt, QTimer, QEasingCurve, QPropertyAnimation, QPoint
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
import pyqtgraph as pg
from pdf_handler import PDFHandler
import os
import time
import random
import numpy as np
import json
import pandas as pd
import ast
import webbrowser
from PyQt5.QtGui import QIcon

class Data:
    def __init__(self):
        self.data = None
        settings = self.load_settings()
        self.roll_no = settings["roll_no"]
        self.path = settings["path"]
        self.first_launch = self.get_first_launch()

    def get_accuracy(self, roll_no):
        # Find the student's data by roll number
        student_data = self.data[self.data['roll_no'] == roll_no].sort_values(by='exam_no')
        
        if student_data.empty:
            return None
            
        accuracies = {
            'physics': [],
            'maths': [],
            'chemistry': [], 
            'all': []
        }
        
        # Get all unique exam numbers for this student
        exam_numbers = student_data['exam_no'].unique()

        # For each exam number
        for exam_no in exam_numbers:
            # Get data for this student and exam
            exam_data = student_data[student_data['exam_no'] == exam_no]
            
            # For each subject
            for subject in ['physics', 'maths', 'chemistry']:
                if not exam_data.empty:
                    # Convert string representation of list to actual list
                    scores = eval(exam_data[subject].iloc[0])
                    
                    total_attempted = scores[0] + scores[1]  # Correct + Wrong
                    if total_attempted > 0:
                        accuracy = (scores[0] / total_attempted) * 100
                    else:
                        accuracy = 0
                else:
                    accuracy = 0
                    
                accuracies[subject].append(round(accuracy, 2))
                
        accuracies['all'] = [(x + y + z)/3 for x, y, z in zip(accuracies['physics'], accuracies['maths'], accuracies['chemistry'])]
        return accuracies
    def get_persons_marks_every_exam(self, roll_no):

        subjects = ['physics', 'maths', 'chemistry']
        filtered_df = self.data[self.data['roll_no'] == roll_no].sort_values(by='exam_no')
        filtered_df = filtered_df[['exam_no', 'exam_type', *subjects]]
        for subject in subjects:
            filtered_df[subject] = filtered_df[subject].apply(ast.literal_eval)
            filtered_df[subject] = filtered_df[subject].apply(lambda x: x[3])
            filtered_df[f"{subject}_old"] = filtered_df.apply(lambda x: x[subject], axis=1)
            filtered_df[subject] = filtered_df.apply(lambda x: x[subject] if x['exam_type'] == 'JEE' else x[subject], axis=1)
            filtered_df[subject] = filtered_df.apply(lambda x: x[subject] *5/2 if x['exam_type'] == 'EAMCET' and (subject == 'physics' or subject == 'chemistry') else x[subject], axis=1)
            filtered_df[subject] = filtered_df.apply(lambda x: x[subject] *5/4 if x['exam_type'] == 'EAMCET' and subject == 'maths' else x[subject], axis=1)

        marks = {
            'physics': filtered_df['physics'].tolist(),
            'maths': filtered_df['maths'].tolist(), 
            'chemistry': filtered_df['chemistry'].tolist()
        }
        
        # Calculate all marks based on exam type
        all_marks = []
        for i in range(len(marks['physics'])):
            exam_type = filtered_df.iloc[i]['exam_type']
            if exam_type == 'JEE':
                # For JEE, simple average of all subjects
                total = (marks['physics'][i] + marks['maths'][i] + marks['chemistry'][i])/3
            else:  # EAMCET
                # For EAMCET, total marks out of 160
                total = (filtered_df.iloc[i]['physics_old'] + filtered_df.iloc[i]['maths_old'] + filtered_df.iloc[i]['chemistry_old'])/160*100
            all_marks.append(total)
            
        marks['all'] = all_marks
        return marks
  
    def get_attempts(self, roll_no):
        student_data = self.data[self.data['roll_no'] == roll_no].sort_values(by='exam_no')
            
        if student_data.empty:
            return None
        
        attempt_rates = {
            'physics': [],
            'maths': [],
            'chemistry': [],
            'all': []
        }
        
        # Get all unique exam numbers for this student
        exam_numbers = student_data['exam_no'].unique()
    
        for exam_no in exam_numbers:
            # Get data for this student and exam
            exam_data = student_data[student_data['exam_no'] == exam_no]
            
            # For each subject
            for subject in ['physics', 'maths', 'chemistry']:
                if not exam_data.empty:
                    # Convert string representation of list to actual list
                    scores = eval(exam_data[subject].iloc[0])
                    
                    attempt_rate = ((scores[0] + scores[1]) / (scores[2] + scores[0] + scores[1])) * 100
                else:
                    attempt_rate = 0
                attempt_rates[subject].append(round(attempt_rate, 2))
                
        attempt_rates['all'] = [(x + y + z)/3 for x, y, z in zip(attempt_rates['physics'], attempt_rates['maths'], attempt_rates['chemistry'])]
                
        return attempt_rates

    def get_roll_no(self):
        return int(self.roll_no)
    
    def get_path(self):
        return self.path
        
    def get_total_exams(self, roll_no):
        student_data = self.data[self.data['roll_no'] == roll_no].sort_values(by='exam_no')
        return sorted(student_data['exam_no'].unique().tolist())

    def get_rank(self, roll_no, exam_no):
        # Get the data for the given exam number
        exam_data = self.data[self.data['exam_no'] == exam_no]
        
        if roll_no not in exam_data['roll_no'].values:
            return None  # Roll number not found
        
        # Get the index of the student in the current exam
        try:
            roll_index = exam_data[exam_data['roll_no'] == roll_no].index[0]
            rank = sum(1 for idx in exam_data.index if idx < roll_index)
        except:
            return None
        
        return max(1, round(rank))  # Rank is 1 + number of students with higher indices

    def get_leaderboard(self, exam_numbers):
        leaderboard = {}
        for roll_no in self.data['roll_no'].unique():
            average_rank = 0
            valid_rank_count = 0  # Track the number of valid ranks
            
            for exam_no in exam_numbers:
                rank = self.get_rank(roll_no, exam_no)
                if rank is not None:
                    average_rank += rank
                    valid_rank_count += 1  # Increment valid rank count
                
            if valid_rank_count > 0:  # Check if there are valid ranks
                average_rank /= valid_rank_count
            else:
                average_rank = len(self.data['roll_no'].unique())  # Assign total count if no valid ranks
            
            leaderboard[f"{int(roll_no)}"] = round(average_rank)
        
        return dict(sorted(leaderboard.items(), key=lambda x: x[1]))
    
    def get_first_launch(self):
        settings = self.load_settings()
        return settings.get("first_launch", True)

    def set_first_launch(self, value):
        settings = self.load_settings()
        settings["first_launch"] = value
        with open("settings.json", "w") as file:
            json.dump(settings, file)

    def load_data(self):
        with open("exam_data.csv", "r") as file:
            data = pd.read_csv(file)
        return data
    
    def save_data(self):
        self.data.to_csv("exam_data.csv", index=False)
        
    def get_names(self):
        names_data = []
        for roll_no, name in zip(self.data['roll_no'], self.data['name']):
            names_data.append([str(int(roll_no)), name])
        return names_data
    
    def save_settings(self, roll_no, path):
        self.roll_no = roll_no
        self.path = path
        with open("settings.json", "w") as file:
            json.dump({"roll_no": roll_no, "path": path}, file)

    def load_settings(self):
        with open("settings.json", "r") as file:
            settings = json.loads(file.read())
        return settings
    
    def check_pdfs(self):
        pdfs_added = 0
        total_pdfs_to_add = len(os.listdir(self.path))
        total_pdfs_to_add -= len(self.get_exam_nos())
        for file in os.listdir(self.path):
            if file.endswith(".pdf") and ("JEEMAIN" in file or "EAMCET" in file):
                if not int(file[8:11]) in self.get_exam_nos():
                    p = PDFHandler(self.path + "/" + file)
                    raw_text = p.cleanse_pdf()
                    if raw_text == -1:
                        continue
                    parsed_data = p.parse_pdf(raw_text)
                    
                    # Convert parsed data to DataFrame format
                    exam_no = file[8:11]
                    exam_type = "JEE" if "JEEMAIN" in file else "EAMCET"
                    
                    new_rows = []
                    for roll_no, student in parsed_data.items():
                        if roll_no != "0":  # Skip the exam type entry
                            new_rows.append({
                                "exam_no": int(exam_no),
                                "exam_type": exam_type, 
                                "roll_no": roll_no,
                                "name": student["name"],
                                "physics": str(student["physics"]),
                                "maths": str(student["maths"]), 
                                "chemistry": str(student["chemistry"])
                            })
                    
                    # Convert to DataFrame and append to existing data
                    new_df = pd.DataFrame(new_rows)
                    self.data = pd.concat([self.data, new_df], ignore_index=True)
                    pdfs_added += 1
                    self.view.update_bottom_text(pdfs_added, total_pdfs_to_add)

        if pdfs_added:
            self.save_data()
            self.view.restart_app()
    def get_exam_nos(self):
        f = [int(exam_no) for exam_no in self.data['exam_no'].unique()]
        f.sort()
        return f

global data
data = Data()

class LineChartWidget(QWidget):
    def __init__(self, data_x, data_y):
        super().__init__()
        
        # Layout with margins for better spacing
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        self.setLayout(layout)

        # PyQtGraph plot widget with modern styling
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setBackground('#1e1e1e')  # Dark gray background
        self.plot_widget.getAxis('left').setPen(color='#888888')
        self.plot_widget.getAxis('bottom').setPen(color='#888888')
        
        # Set fixed y range based on target data
        y_min = 0
        y_max = max(data_y) + 10  # Add padding
        self.plot_widget.setYRange(y_min, y_max, padding=0)
        
        layout.addWidget(self.plot_widget)

        # Store target data
        self.target_x = data_x
        self.target_y = data_y
        
        # Initialize current data starting from 0
        self.current_x = self.target_x.copy()
        self.current_y = [0] * len(self.target_y)  # Start all points at y=0
        
        # Create plot item with modern styling
        self.plot_item = self.plot_widget.plot(self.current_x, self.current_y,
                                             pen=pg.mkPen(color='#00bfff', width=3),
                                             symbol='o',
                                             symbolSize=5,
                                             symbolBrush='#1f9bd9',
                                             symbolPen='#1f9bd9')

        # Add hover label
        self.hover_label = pg.TextItem(text='', color='#ffffff', anchor=(0, 1))
        self.hover_label.setParentItem(self.plot_widget.plotItem)
        
        # Add hover line
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='#888888')
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        
        # Connect mouse move event
        self.plot_widget.scene().sigMouseMoved.connect(self.onMouseMoved)

        # Setup timer for smooth animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10)  # ~30 FPS for smoother animation
        
        # Animation parameters
        self.animation_steps = 60  # More steps for smoother animation
        self.current_step = 0
        
        # Initialize start values
        self.start_x = self.current_x.copy()
        self.start_y = self.current_y.copy()
        
        # Animation state
        self.animating_down = True  # First animate down to 0
        self.new_x = None  # Store new values during animation
        self.new_y = None

    def onMouseMoved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            
            # Snap to nearest integer x value
            x_val = round(x_val)
            
            # Find corresponding y value at this x position
            try:
                idx = self.current_x.index(x_val)
                y_val = self.current_y[idx]
            except (ValueError, IndexError):
                return
                
            # Position label in top right corner
            view_box = self.plot_widget.getViewBox()
            view_range = view_box.viewRange()
            label_x = 30 # Use left edge of view range
            label_y = 30  # Keep top edge for y position
            
            # Update hover label and line
            self.hover_label.setText(f"Exam number: {x_val}\nValue: {y_val:.1f}")
            self.hover_label.setPos(label_x, label_y)
            self.vLine.setPos(x_val)
            
            # Show items
            self.hover_label.show()
            self.vLine.show()
        else:
            # Hide items when mouse leaves plot
            self.hover_label.hide()
            self.vLine.hide()

    def update_data_points(self, new_x, new_y):
        """Update the plot with new data points and animate the transition"""
        # Store new target values
        self.new_x = new_x
        self.new_y = new_y
        
        # Start animation down to 0 first
        self.start_x = self.current_x.copy()
        self.start_y = self.current_y.copy()
        self.target_y = [0] * len(self.current_y)  # Animate to 0
        self.target_x = self.current_x.copy()  # Keep x positions same
        
        self.animating_down = True
        self.current_step = 0
        
        # Update y-axis range for new data
        y_min = 0
        y_max = max(new_y) + 10
        self.plot_widget.setYRange(y_min, y_max, padding=0)
        
        # Restart animation timer if not running
        if not self.timer.isActive():
            self.timer.start(10)
            
    def update_plot(self):
        if self.current_step < self.animation_steps:
            # Calculate intermediate values using smooth easing
            progress = self.current_step / self.animation_steps
            ease = 0.5 - np.cos(progress * np.pi) / 2  # Smooth easing function
            
            # Animate each point from start to target value
            try:
                for i in range(len(self.current_y)):
                    self.current_y[i] = self.start_y[i] + (self.target_y[i] - self.start_y[i]) * ease
                    self.current_x[i] = self.start_x[i] + (self.target_x[i] - self.start_x[i]) * ease
            except:
                pass
            
            self.plot_item.setData(self.current_x, self.current_y)
            self.current_step += 2
        else:
            if self.animating_down:
                
                # Switch to animating up to new values
                self.animating_down = False
                self.current_step = 0
                if self.new_x is not None and self.new_y is not None:
                    self.current_x = self.new_x.copy()
                    self.current_y = [0] * len(self.new_y)
                self.start_x = self.current_x.copy()
                self.start_y = self.current_y.copy()
                self.target_x = self.new_x
                self.target_y = self.new_y
            else:
                # Animation complete, stop the timer
                self.timer.stop()
class SelfAnalysisView(QWidget):
    def __init__(self):
        super().__init__()
        
        # Main layout setup
        layout = QVBoxLayout()
        layout.setDirection(QVBoxLayout.LeftToRight)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        self.setLayout(layout)

        self.metric = "Accuracy"
        self.subject = "physics"
        self.roll_no = data.get_roll_no()

        # Initialize graph with physics data
        self.graph = LineChartWidget(data.get_total_exams(self.roll_no), data.get_accuracy(self.roll_no)[self.subject])
        layout.addWidget(self.graph)

        # Right side container setup
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setDirection(QVBoxLayout.TopToBottom)
        right_layout.setContentsMargins(0, 0, 0, 0)  # Removed margins
        right_layout.setSpacing(10)  # Reduced spacing
        right_container.setLayout(right_layout)
        
        # Title setup
        title_label = QLabel("Self Analysis")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
        """)
        
        # Dropdown setup
        self.subject_combo = QComboBox()
        self.subject_combo.addItems(['Physics', 'Chemistry', 'Maths', 'All'])
        self.subject_combo.currentTextChanged.connect(self.update_graph)
        self.subject_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2d2d2d;
                min-width: 150px;
                max-width: 300px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 8px;
            }
            QComboBox:hover {
                border-color: #4d4d4d;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 2px solid #3d3d3d;
                selection-background-color: #3d3d3d;
                outline: none;
            }
        """)
        # Radio button group for metric selection
        self.metric_group = QWidget()
        metric_layout = QHBoxLayout()
        metric_layout.setDirection(QHBoxLayout.TopToBottom)
        metric_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        metric_layout.setSpacing(10)  # Reduced spacing
        self.metric_group.setLayout(metric_layout)
        
        self.accuracy_radio = QRadioButton("Accuracy %")
        self.attempts_radio = QRadioButton("Attempt %") 
        self.marks_radio = QRadioButton("Marks")
        self.accuracy_radio.clicked.connect(lambda: self.change_metric("Accuracy"))
        self.attempts_radio.clicked.connect(lambda: self.change_metric("Attempts"))
        self.marks_radio.clicked.connect(lambda: self.change_metric("Marks"))
        self.accuracy_radio.setChecked(True)
        
        # Style the radio buttons
        radio_style = """
            QRadioButton {
                spacing: 8px;
                padding: 4px;
                min-width: 150px;
                max-width: 300px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #3d3d3d;
                border-radius: 9px;
                background-color: #2d2d2d;
            }
            QRadioButton::indicator:checked {
                background-color: #31588c;
                border: 2px solid #5d5d5d;
            }
            QRadioButton:hover {
                color: #cccccc;
            }
        """
        self.accuracy_radio.setStyleSheet(radio_style)
        self.attempts_radio.setStyleSheet(radio_style)
        self.marks_radio.setStyleSheet(radio_style)
        
        metric_layout.addWidget(self.accuracy_radio)
        metric_layout.addWidget(self.attempts_radio)
        metric_layout.addWidget(self.marks_radio)
        metric_layout.setAlignment(self.accuracy_radio, Qt.AlignmentFlag.AlignLeft)
        metric_layout.setAlignment(self.attempts_radio, Qt.AlignmentFlag.AlignLeft)
        metric_layout.setAlignment(self.marks_radio, Qt.AlignmentFlag.AlignLeft)
        average_accuracy = round(sum(data.get_accuracy(self.roll_no)[self.subject])/len(data.get_total_exams(self.roll_no)), 2)
        average_attempts = round(sum(data.get_attempts(self.roll_no)[self.subject])/len(data.get_total_exams(self.roll_no)), 2)

        self.accuracy_label = QLabel(f"Average Accuracy: {average_accuracy}%")
        self.attempts_label = QLabel(f"Average Attempts: {average_attempts}%")
        self.accuracy_label.setStyleSheet("""
            font-size: 16px;
            color: #e0e0e0;
            padding: 8px 12px;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            background-color: #2d2d2d;
            margin: 4px;
            min-width: 200px;
            max-width: 300px;
        """)
        self.attempts_label.setStyleSheet("""
            font-size: 16px;
            color: #e0e0e0;
            padding: 8px 12px;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            background-color: #2d2d2d;
            margin: 4px;
            min-width: 200px;
            max-width: 300px;
        """)

        # Roll number selection
        roll_label = QLabel("Select Roll Number:")
        roll_label.setStyleSheet("""
            font-size: 16px;
            color: #e0e0e0;
            margin-bottom: 4px;
        """)
        
        self.roll_input = QLineEdit()
        self.roll_input.setStyleSheet("""
            QLineEdit {
                font-size: 14px;
                color: #e0e0e0;
                padding: 8px 12px;
                border: 2px solid #3d3d3d;
                border-radius: 6px;
                background-color: #2d2d2d;
                min-width: 200px;
                max-width: 300px;
            }
            QLineEdit:focus {
                border: 2px solid #4d4d4d;
            }
        """)
        
        # Set placeholder text
        self.roll_input.setPlaceholderText("Enter roll number...")
        
        # Connect roll number input to handler
        self.roll_input.returnPressed.connect(self.handle_roll_change)
        
        # Set current roll number
        self.roll_input.setText(str(self.roll_no))

        # Create labels for deviation stats
        self.deviation_label = QLabel()
        self.deviation_label.setStyleSheet("""
            font-size: 16px;
            color: #e0e0e0;
            padding: 8px 12px;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            background-color: #2d2d2d;
            margin-top: 30px;
            min-width: 200px;
            max-width: 300px;
        """)

        
        # Initial calculation
        self.calculate_deviation()
        
        # Connect to update when data changes

        # Add widgets to right layout with reduced spacing
        right_layout.addWidget(title_label)
        right_layout.addSpacing(5)  # Small gap after title
        right_layout.addWidget(self.subject_combo)
        right_layout.addSpacing(5)  # Small gap after combo box
        right_layout.addWidget(self.metric_group)
        right_layout.addWidget(self.accuracy_label)
        right_layout.addWidget(self.attempts_label)
        right_layout.addWidget(roll_label)
        right_layout.addWidget(self.roll_input)

        right_layout.addStretch()  # Push everything up
        right_layout.addWidget(self.deviation_label)


        # Add right container to main layout
        layout.addWidget(right_container)
        # Set size policy for right container to match graph
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.setStretch(0, 2)  # Graph container
        layout.setStretch(1, 1)  # Right container

    def handle_roll_change(self):

        try:
            new_roll = int(self.roll_input.text())
            names_data = data.get_names()
            roll_numbers = [int(roll[0]) for roll in sorted(names_data)]
            
            if new_roll not in roll_numbers:
                self.display_error("Roll number not found")
                self.roll_input.setText(str(self.roll_no))  # Reset to previous valid roll
                return
                
            self.roll_no = new_roll
            self.update_graph(self.subject_combo.currentText())
            
        except ValueError:
            self.display_error("Please enter a valid roll number")
            self.roll_input.setText(str(self.roll_no))  # Reset to previous valid roll

        self.calculate_deviation()
        

    def display_error(self, message):
        # Create QMessageBox for error display
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(f"⚠️  Error: {message}")
        error_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 16px;
                min-width: 300px;
                padding: 20px;
            }
            QMessageBox QPushButton {
                background-color: #FF3333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #FF4444;
            }
        """)
        error_box.exec_()
    
    def calculate_deviation(self):
            # Get accuracy and attempts data
            accuracies = data.get_accuracy(self.roll_no)
            attempts = data.get_attempts(self.roll_no)
            
            if not accuracies or not attempts:
                self.deviation_label.setText("No data available")
                return
                
            # Get latest values
            latest_acc = accuracies['all'][-1]
            latest_att = attempts['all'][-1]
            
            # Calculate means excluding latest value
            mean_acc = sum(accuracies['all'][:-1]) / len(accuracies['all'][:-1]) if len(accuracies['all']) > 1 else latest_acc
            mean_att = sum(attempts['all'][:-1]) / len(attempts['all'][:-1]) if len(attempts['all']) > 1 else latest_att
            
            # Calculate deviations
            acc_dev = latest_acc - mean_acc
            att_dev = latest_att - mean_att
            
            # Format text with arrows and colors
            acc_color = "#00FF00" if acc_dev > 0 else "#FF0000" if acc_dev < 0 else "white"
            att_color = "#00FF00" if att_dev > 0 else "#FF0000" if att_dev < 0 else "white"
            
            acc_text = f"↑ +{acc_dev:.1f}%" if acc_dev > 0 else f"↓ {acc_dev:.1f}%" if acc_dev < 0 else "→ 0%"
            att_text = f"↑ +{att_dev:.1f}%" if att_dev > 0 else f"↓ {att_dev:.1f}%" if att_dev < 0 else "→ 0%"
            
            self.deviation_label.setText(f"Deviation from mean (latest exam):<br><br>Accuracy: <span style='color:{acc_color}'>{acc_text}</span><br>Attempt Rate: <span style='color:{att_color}'>{att_text}</span>")
            self.deviation_label.setTextFormat(Qt.RichText)
            
    def update_graph(self, subject):
        subject = subject.lower()
        self.subject = subject

        average_accuracy = round(sum(data.get_accuracy(self.roll_no)[self.subject])/len(data.get_total_exams(self.roll_no)), 2)
        average_attempts = round(sum(data.get_attempts(self.roll_no)[self.subject])/len(data.get_total_exams(self.roll_no)), 2)

        self.accuracy_label.setText(f"Average Accuracy: {average_accuracy}%")
        self.attempts_label.setText(f"Average Attempts: {average_attempts}%")

        data_to_plot = (data.get_accuracy(self.roll_no)[subject] if self.metric == "Accuracy" 
                        else data.get_attempts(self.roll_no)[subject] if self.metric == "Attempts" 
                        else data.get_persons_marks_every_exam(self.roll_no)[subject] if self.metric == "Marks" 
                        else [])
        self.graph.update_data_points(data.get_total_exams(self.roll_no), data_to_plot)

    def change_metric(self, metric):
        if metric == "Accuracy":
            self.graph.update_data_points(data.get_total_exams(self.roll_no), data.get_accuracy(self.roll_no)[self.subject])
            self.metric = "Accuracy"
        elif metric == "Attempts":
            self.graph.update_data_points(data.get_total_exams(self.roll_no), data.get_attempts(self.roll_no)[self.subject])
            self.metric = "Attempts"
        elif metric == "Marks":
            self.graph.update_data_points(data.get_total_exams(self.roll_no), data.get_persons_marks_every_exam(self.roll_no)[self.subject])
            self.metric = "Marks"

class LeaderboardView(QWidget):
    def __init__(self):
        super().__init__()
        self.exam_numbers = data.get_exam_nos()
        self.leaderboard_data = data.get_leaderboard(self.exam_numbers)

        # Get names data from exam data
        self.names_data = {str(row['roll_no']): row['name'].strip() 
                          for _, row in data.data.iterrows()}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.roll_no = data.get_roll_no()

        # Title label with modern styling and tooltip
        title_layout = QHBoxLayout()

        title = QLabel("Leaderboard")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
        """)
        title.setToolTip("<span style='color: #31588c;'>Shows average ranks (Top nth) across all exams.</span>")
        title_layout.addWidget(title, alignment=Qt.AlignCenter)
        
        self.selection_box = QLineEdit()
        self.selection_box.setPlaceholderText("Filters")
        self.selection_box.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: #2d2d2d;
            padding: 10px;  /* Added padding for consistency */
            border: 1px solid rgba(255, 255, 255, 0.2);  /* Added border for a more defined look */
            border-radius: 5px;  /* Added border radius for rounded corners */
        """)
        self.selection_box.returnPressed.connect(self.regenerate_leaderboard)
        self.path_input_help = QLabel("?")
        self.path_input_help = QPushButton("?")  # Change to QPushButton
        self.path_input_help.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: #31588c;  /* Button background color */
            padding: 10px;  /* Added padding for a button-like appearance */
            border: none;  /* No border for a cleaner look */
            border-radius: 5px;  /* Rounded corners */
        """)
        self.path_input_help.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Ensure it can receive mouse events
        self.path_input_help.setStyleSheet(self.path_input_help.styleSheet() + """
                background: #4a7fba;  /* Darker shade on hover */
        """)
        msg = QMessageBox(self)
        msg.setWindowTitle("Filter Help")
        msg.setText("There are two ways to filter the leaderboard:\n\n"
                   "1. Enter exam numbers separated by commas to filter the leaderboard.\n\n"
                   "Examples:\n"
                   "• 15                     -> Show only exam 15\n"
                   "• 15,20 (or) 15 20       -> Show exams 15 and 20\n\n"
                   "2. Enter a range of exam numbers to filter the leaderboard.\n\n"
                   "Examples:\n"
                   "• from 15 till 20        -> Show exams 15 to 20 including 20\n"
                   "• from 15 to 20          -> Show exams 15 to 20 excluding 20\n"
                   "• from 15 to 20 +25 +28  -> Show exams 15 to 20 and 25 and 28\n"
                   "• from 15 to 20 -18      -> Show exams 15 to 20 excluding 18\n"
                   "• Note: You can only use any one of the two methods at a time.\n"
                   "• Note: try to put +20 before -18 if using both + and - at the same time (no space between + and 20)\n"
                   "• Tip: You can hover over the filter input to see the current exam numbers being used to generate the leaderboard.\n"
                   "• Leave empty to show all exams")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Consolas;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-family: Consolas;
            }
            QPushButton {
                background-color: #31588c;
                color: #ffffff;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-family: Consolas;
            }
            QPushButton:hover {
                background-color: #4a7fba;
            }
        """)
        self.path_input_help.clicked.connect(lambda: msg.exec_())

        # Show current exam numbers on hover
        self.selection_box.setToolTip(", ".join(map(str, self.exam_numbers)))
        
        title_layout.addWidget(self.selection_box)
        title_layout.addWidget(self.path_input_help)
        self.layout.addLayout(title_layout)

        # Create a scroll area with dark theme
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                background-clip: padding-box;
            }
            QScrollArea QScrollBar:vertical {
                width: 12px;
                background: #2d2d2d;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollArea QScrollBar::handle:vertical {
                background: #31588c;  /* Changed active scroll bar color */
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollArea QScrollBar::add-line:vertical,
            QScrollArea QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollArea QScrollBar::add-page:vertical,
            QScrollArea QScrollBar::sub-page:vertical {
                background: none;
            }
            QWidget {
                background-color: #1e1e1e;
            }
        """)
        # Create a widget to hold the leaderboard items
        self.leaderboard_widget = QWidget()
        self.leaderboard_layout = QVBoxLayout(self.leaderboard_widget)
        self.leaderboard_layout.setSpacing(8)
        first_rank = list(self.leaderboard_data.items())[0]
        
        # First place label
        # Create labels for name and rank
        name_label = QLabel(f"{first_rank[0]} - {self.names_data.get(first_rank[0], 'Unknown')}")
        rank_label = QLabel(f"{first_rank[1]}")
        
        # Create horizontal layout
        first_place_layout = QHBoxLayout()
        first_place_layout.addWidget(name_label)
        first_place_layout.addStretch()
        first_place_layout.addWidget(rank_label)
        
        # Create container widget
        first_place_container = QWidget()
        first_place_container.setLayout(first_place_layout)
        first_place_container.setStyleSheet("""
            background-color: #2d2d2d;
            padding: 12px;
            border-radius: 4px;
        """)
        
        # Style the labels
        name_label.setStyleSheet("""
            font-size: 28px;
            color: #31588c;
            background: transparent;
            font-weight: bold;
        """)
        rank_label.setStyleSheet("""
            font-size: 28px;
            color: #31588c;
            background: transparent;
            font-weight: bold;
        """)
        
        self.leaderboard_layout.addWidget(first_place_container)

        for roll_no, average_rank in list(self.leaderboard_data.items())[1:]:
            name = self.names_data.get(roll_no, "Unknown")
            label = QLabel(f"{roll_no} - {name}")
            rank_label = QLabel(f"{average_rank}")
            
            # Create a horizontal layout for each row
            row_layout = QHBoxLayout()
            row_layout.addWidget(label)
            row_layout.addStretch()
            row_layout.addWidget(rank_label)
            
            # Create a container widget for the row
            container = QWidget()
            container.setLayout(row_layout)
            container.setStyleSheet("""
                background-color: #2d2d2d;
                padding: 10px;
                border-radius: 4px;
            """)
            
            # Style the labels
            label.setStyleSheet("""
                font-size: 16px;
                color: #ffffff;
                background: transparent;
            """)
            
            if int(roll_no) == self.roll_no:
                container.setStyleSheet("""
                    background-color: #31588c;
                    padding: 10px;
                    border-radius: 4px;
                """)
            
            self.leaderboard_layout.addWidget(container)

        # Set the leaderboard widget in the scroll area
        self.scroll_area.setWidget(self.leaderboard_widget)

        # Add the scroll area to the main layout
        self.layout.addWidget(self.scroll_area)
        self.layout.setContentsMargins(20, 20, 20, 20)

    def regenerate_leaderboard(self):
        # Clear all widgets at once
        while self.leaderboard_layout.count():
            item = self.leaderboard_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        # Create loading screen
        loading_sign = QLabel("Crunching numbers...\n This may take a while...")
        loading_sign.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: transparent;
            margin-top: 20px;
        """)
        loading_sign.setAlignment(Qt.AlignCenter)
        self.leaderboard_layout.addWidget(loading_sign)
        self.leaderboard_widget.update()
        QApplication.processEvents()  # Force UI update
        
        # Add a small delay to ensure loading screen is visible
        QTimer.singleShot(100, self._process_leaderboard)

    def _process_leaderboard(self):
        # Process text input
        text = self.selection_box.text()
        possible_exam_numbers = data.get_exam_nos()
        self.exam_numbers = []

        stuff_to_remove = []
        stuff_to_add = []
        if text.replace(",", " ").replace(" ", "").isdigit():
            self.exam_numbers = [int(num.strip().replace(",", "")) if num != "" else -1 for num in text.replace(",", " ").split(" ")] 
            self.exam_numbers = list(set([num for num in self.exam_numbers if num != -1]))
            self.exam_numbers = [num for num in self.exam_numbers if num in possible_exam_numbers]
        else:
            start_index = data.get_exam_nos()[0]
            end_index = data.get_exam_nos()[-1]
            try:
                if "from" in text:
                    start_index = int(text.strip().split("from")[1].strip().split(" ")[0])
                if "till" in text:
                    end_index = int(text.strip().split("till")[1].strip().split(" ")[0]) - 1    
                if "to" in text:
                    end_index = int(text.strip().split("to")[1].strip().split(" ")[0])
                
                subtractions = [int(part.strip()) for part in text.split("-")[1:] if part.strip().isdigit()]
                stuff_to_remove.extend(subtractions)
                
                additions = [int(part.strip()) for part in text.split("+")[1:] if part.strip().isdigit()]
                stuff_to_add.extend(additions)
            except:
                pass

            self.exam_numbers = list(set(range(start_index, end_index + 1)))
            self.exam_numbers = [num for num in (self.exam_numbers + stuff_to_add)]
            self.exam_numbers = [num for num in self.exam_numbers if num not in stuff_to_remove]
            self.exam_numbers = [num for num in self.exam_numbers if num in possible_exam_numbers]

        self.leaderboard_data = data.get_leaderboard(self.exam_numbers)
        # Create the leaderboard layout
        self.leaderboard_layout = QVBoxLayout()
        self.leaderboard_widget = QWidget()
        self.leaderboard_widget.setLayout(self.leaderboard_layout)
        self.scroll_area.setWidget(self.leaderboard_widget)
        self.layout.addWidget(self.scroll_area)

        # Get first place data
        first_rank = list(self.leaderboard_data.items())[0]
        self.selection_box.setToolTip(", ".join(map(str, self.exam_numbers)))
        
        # Create first place labels
        name_label = QLabel(f"{first_rank[0]} - {self.names_data.get(first_rank[0], 'Unknown')}")
        rank_label = QLabel(f"{first_rank[1]}")
        
        # Create layout for first place
        first_place_layout = QHBoxLayout()
        first_place_layout.addWidget(name_label)
        first_place_layout.addStretch()
        first_place_layout.addWidget(rank_label)
        
        # Create container widget
        first_place_container = QWidget()
        first_place_container.setLayout(first_place_layout)
        first_place_container.setStyleSheet("""
            background-color: #2d2d2d;
            padding: 12px;
            border-radius: 4px;
        """)
        
        # Style the labels
        name_label.setStyleSheet("""
            font-size: 28px;
            color: #31588c;
            background: transparent;
            font-weight: bold;
        """)
        rank_label.setStyleSheet("""
            font-size: 28px;
            color: #31588c;
            background: transparent;
            font-weight: bold;
        """)

        # Add first place to layout
        self.leaderboard_layout.addWidget(first_place_container)

        # Add remaining leaderboard entries
        for roll_no, rank in list(self.leaderboard_data.items())[1:]:
            entry_layout = QHBoxLayout()
            
            name_label = QLabel(f"{roll_no} - {self.names_data.get(roll_no, 'Unknown')}")
            rank_label = QLabel(f"{rank}")
            
            entry_layout.addWidget(name_label)
            entry_layout.addStretch()
            entry_layout.addWidget(rank_label)
            
            entry_container = QWidget()
            entry_container.setLayout(entry_layout)
            
            # Highlight current user's entry
            if roll_no == str(self.roll_no):
                entry_container.setStyleSheet("""
                    background-color: #31588c;
                    padding: 8px;
                    border-radius: 4px;
                    margin: 2px;
                """)
                name_label.setStyleSheet("""
                    font-size: 16px;
                    color: white;
                    background: transparent;
                """)
                rank_label.setStyleSheet("""
                    font-size: 16px;
                    color: white;
                    background: transparent;
                """)
            else:
                entry_container.setStyleSheet("""
                    background-color: #2d2d2d;
                    padding: 8px;
                    border-radius: 4px;
                    margin: 2px;
                """)
                name_label.setStyleSheet("""
                    font-size: 16px;
                    color: #ffffff;
                    background: transparent;
                """)
                rank_label.setStyleSheet("""
                    font-size: 16px;
                    color: #ffffff;
                    background: transparent;
                """)
            
            self.leaderboard_layout.addWidget(entry_container)           

class SettingsView(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        title = QLabel("Settings")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
        """)
        self.layout.addWidget(title, alignment=Qt.AlignLeft)

        # Create a uniform size for input fields
        input_width = 400  # Set a consistent width for all input fields

        roll_no_layout = QHBoxLayout()
        roll_no_label = QLabel("Roll Number:")
        roll_no_label.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: transparent;
            max-width: 100px;
        """)
        self.roll_no_input = QLineEdit() 
        self.roll_no_input.setText(str(data.get_roll_no()))
        self.roll_no_input.setPlaceholderText("Enter your roll number")  # Set hint text
        self.roll_no_input.setStyleSheet(f"""
            font-size: 16px;
            color: #ffffff;
            background: #2d2d2d;  
            border: 2px solid #3d3d3d;  
            border-radius: 4px;  
            padding: 5px;  
            max-width: {input_width}px;  /* Consistent width */
        """)
        roll_no_layout.addWidget(roll_no_label)
        roll_no_layout.addWidget(self.roll_no_input)
        roll_no_tooltip = QLabel("This is your roll number, not your name. This roll number will be used to identify you in the app.")
        roll_no_tooltip.setStyleSheet("""
            font-size: 12px;
            color: #888888;
            background: transparent;
            margin-bottom: 35px;
        """)


        path_layout = QHBoxLayout()
        path_label = QLabel("Path:")
        path_label.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: transparent;
            max-width: 100px;
        """)
        self.path_input = QLineEdit()
        self.path_input.setText(data.get_path())
        self.path_input.setStyleSheet(f"""
            font-size: 16px;
            color: #ffffff;
            background: #2d2d2d;  
            border: 2px solid #3d3d3d;  
            border-radius: 4px;  
            padding: 5px;  
            max-width: {input_width}px;  /* Consistent width */
        """)
        self.path_input.setPlaceholderText("Enter the path to your marks folder")  # Set hint text
       
        path_tooltip = QLabel("This is the path to your marks folder. It will be checked for new marks pdfs auto-magically. \nCurrently only supports JEE mains and Eamcet pdfs.")
        path_tooltip.setStyleSheet("""
            font-size: 12px;
            color: #888888;
            background: transparent;
            margin-bottom: 35px;
        """)
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)

        save_button = QPushButton("Save")
        save_button.setStyleSheet("""
            QPushButton {
            font-size: 16px;
            color: #ffffff;
            background: #31588c;
            border: none;
            border-radius: 4px;
            width: 100px;
            height: 30px;
        }
        QPushButton:hover {
            background: #3e6dad;
        }
        """)
        save_button.clicked.connect(self.save_settings)

        open_source_layout = QHBoxLayout()
        open_source_label = QLabel("I am open source! Check out the source code here:")
        open_source_label.setStyleSheet("""
            font-size: 16px;
            color: #ffffff;
            background: transparent;
        """)
        open_source_button = QPushButton("Github")
        open_source_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: #888888;  /* Grey color */
                border: none;
                border-radius: 4px;
                width: 100px;
                height: 30px;
            }
            QPushButton:hover {
                color: #3e6dad;  /* Accent color on hover */
            }
        """)
        # open_source_button.clicked.connect(self.open_source)
        open_source_button.clicked.connect(lambda: webbrowser.open("https://github.com/Vishvesh1542/marks-analysis"))
        open_source_layout.setAlignment(Qt.AlignLeft)
        open_source_layout.addWidget(open_source_label)
        open_source_layout.addWidget(open_source_button)



        self.layout.addLayout(roll_no_layout)
        self.layout.setAlignment(roll_no_layout, Qt.AlignLeft)
        self.layout.addWidget(roll_no_tooltip, alignment=Qt.AlignLeft)

        self.layout.addLayout(path_layout)
        self.layout.setAlignment(path_layout, Qt.AlignLeft)
        self.layout.addWidget(path_tooltip, alignment=Qt.AlignLeft)

        self.layout.addWidget(save_button, alignment=Qt.AlignLeft)

        self.layout.addStretch()
        self.layout.addLayout(open_source_layout)

    def save_settings(self):
        if self.roll_no_input.text().isdigit():
            new_roll = int(self.roll_no_input.text())
            # names_data = data.get_names()
            # roll_numbers = [int(roll[0]) for roll in sorted(names_data)]
            
            # if new_roll not in roll_numbers:
            #     self.display_error("Roll number not found")
            #     return
                
            roll_no = new_roll
            
        else:
            self.display_error("Please enter a valid roll number")
            return
        
        if os.path.exists(self.path_input.text()):
            data.save_settings(roll_no, self.path_input.text())
            self.display_success("Settings saved successfully")
            if self.parent:
                try:
                    data.set_first_launch(False)
                    self.parent.done_onboarding()
                except:
                    pass
            self.parent.restart_app()
            return
            
            
        else:
            self.display_error("Path does not exist")
            self.path_input.setText(data.get_path())  # Reset to previous valid path
            return

    def display_error(self, message):        # Create QMessageBox for error display
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(f"⚠️  Error: {message}")
        error_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 16px;
                min-width: 300px;
                padding: 20px;
            }
            QMessageBox QPushButton {
                background-color: #FF3333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #FF4444;
            }
        """)
        error_box.exec_()

    def display_success(self, message):
        success_box = QMessageBox(self)
        success_box.setIcon(QMessageBox.Information)
        success_box.setWindowTitle("Success")
        success_box.setText(f"✅  Success: {message}")

        success_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 16px;
                min-width: 300px;
                padding: 20px;
            }
            QMessageBox QPushButton {
                background-color: #31588c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #3e6dad;
            }
        """)
        success_box.exec_()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marks Analytics")
        self.setGeometry(100, 100, 1200, 800)  # Larger default size

        # Set modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with margins
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)

        # Tab widget with modern styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { 
                border: 0;
                background: #1e1e1e;
                alignment: center;
            }
            QTabWidget::tab-bar { 
                alignment: center;
            }
            QTabBar::tab {
                padding: 6px 14px;
                min-width: 100px;
                margin: 0;
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.6);
                font-size: 12px;
                font-weight: 400;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI';
                text-align: center;  /* Changed to text-align for center alignment */
            }
            QTabBar::tab:selected {
                background: transparent;
                color: #ffffff;
                border-bottom: 1px solid #31588c;
            }
            QTabBar::tab:hover:!selected {
                background: transparent;
                color: rgba(255, 255, 255, 0.8);
            }
            QTabBar::tab:pressed {
                background: transparent;
                color: #ffffff;
            }
        """)
        main_layout.addWidget(self.tab_widget)


        # Add tabs with modern names
        # Remove West position to make tabs appear horizontally (default is North/top)
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Add tabs with icons
        if not data.get_first_launch():
            self.tab_widget.clear()
            splash_screen = QLabel("Loading...")
            splash_screen.setStyleSheet("""
                font-size: 24px;
                color: #ffffff;
                background: transparent;
                qproperty-alignment: AlignCenter;
            """)
            
            splash_container = QWidget()
            splash_layout = QVBoxLayout()
            splash_layout.addStretch()
            splash_layout.addWidget(splash_screen)
            self.bottom_text = QLabel("Calculating your marks...")
            self.bottom_text.setStyleSheet("""
                font-size: 16px;
                color: #aaaaaa;
                background: transparent;
                qproperty-alignment: AlignCenter;
            """)
            splash_layout.addWidget(self.bottom_text)
            splash_layout.addStretch()
            splash_container.setLayout(splash_layout)
            
            # Set background color for splash container
            splash_container.setStyleSheet("""
                background-color: #1e1e1e;
            """)
            
            self.tab_widget.addTab(splash_container, "Loading")
            
            self.tab_widget.show()
            self.tab_widget.update()

            # Use QTimer to delay loading so splash screen appears
            QTimer.singleShot(10, self.load_data_with_splash)


        else:
            settings_icon = QIcon("icons/settings.png")
            settings_view = SettingsView(parent=self)
            settings_view.parent = self
            self.tab_widget.addTab(settings_view, settings_icon, "Settings")
        
    def load_view(self):
        self.tab_widget.clear()
        analytics_icon = QIcon("icons/analytics.png")  # You'll need to add icon files
        self.tab_widget.addTab(SelfAnalysisView(), analytics_icon, "Analytics")
    
        leaderboard_icon = QIcon("icons/leaderboard.png")
        self.tab_widget.addTab(LeaderboardView(), leaderboard_icon, "Leaderboard")
        
        settings_icon = QIcon("icons/settings.png")
        settings_view = SettingsView(parent=self)
        settings_view.parent = self
        self.tab_widget.addTab(settings_view, settings_icon, "Settings")
        

    def done_onboarding(self):
        self.tab_widget.removeTab(0)
        analytics_icon = QIcon("icons/analytics.png")  # You'll need to add icon files
        leaderboard_icon = QIcon("icons/leaderboard.png")
        settings_icon = QIcon("icons/settings.png")

        self.tab_widget.addTab(SelfAnalysisView(), analytics_icon, "Analytics")
        self.tab_widget.addTab(LeaderboardView(), leaderboard_icon, "Leaderboard")
        settings_view = SettingsView(parent=self)
        settings_view.parent = self
        self.tab_widget.addTab(settings_view, settings_icon, "Settings")

    def load_data_with_splash(self):
        QApplication.processEvents()
        data.view = self
        data.data = data.load_data()
        self.check_pdfs()
        data.data = data.load_data()
        if data.data.empty:
            self.display_error("No data found in given path. \nPlease check your settings.")
            data.set_first_launch(True)
            data.save_settings(data.roll_no, data.path)
            self.restart_app()
        if data.data[data.data["roll_no"] == data.roll_no].empty:
            self.display_error("No data found for your roll no. \nPlease check your settings.json \n Click ok to restart the app and try again.")
            self.restart_app()
            return
        self.load_view()

    def restart_app(self):
        self.tab_widget.clear()
        restart_screen = QLabel("Please restart the app")
        restart_screen.setStyleSheet("""
            font-size: 24px;
            color: #ffffff;
            background: transparent;
            qproperty-alignment: AlignCenter;
        """)
        self.tab_widget.addTab(restart_screen, "Restarting")
        QApplication.processEvents()
        
        # Force UI update before restarting
        QApplication.instance().processEvents()
        QApplication.instance().flush()

    def check_pdfs(self):
        data.check_pdfs()

    def display_error(self, message, restart=False):        # Create QMessageBox for error display
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(f"⚠️  Error: {message}")
        error_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 16px;
                min-width: 300px;
                padding: 20px;
            }
            QMessageBox QPushButton {
                background-color: #FF3333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #FF4444;
            }
        """)
        if restart:
            restart_button = QPushButton("Restart")     
            restart_button.setStyleSheet("""
                background-color: #FF3333;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            """)
            error_box.addButton(restart_button, QMessageBox.RejectRole)
            # error_box.setDefaultButton(QMessageBox.RejectRole)
            error_box.buttonClicked.connect(self.restart_app)
        error_box.exec_()

    def update_bottom_text(self, pdf_number, total_pdfs):
        self.bottom_text.setText(f"Processing pdfs... {pdf_number}/{total_pdfs}")
        QApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons/icon.png"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
