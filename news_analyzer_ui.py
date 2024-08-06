from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, Rectangle, Line
from kivy.uix.widget import Widget

class ImageButton(ButtonBehavior, Image):
    pass

class NewsAnalyzerUI(BoxLayout):
    def __init__(self, analyze_callback, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [15, 15, 15, 15]
        self.spacing = 15
        self.analyze_callback = analyze_callback

        self.bind(size=self._update_rect, pos=self._update_rect)

        with self.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # Light gray background
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self._create_title_bar()
        self._create_input_field()
        self._create_analyze_button()
        self._create_results_area()

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def _create_title_bar(self):
        title_layout = BoxLayout(size_hint_y=None, height=60)
        with title_layout.canvas.before:
            Color(0.9, 0.9, 0.9, 1)  # Slightly darker background for title bar
            Rectangle(size=title_layout.size, pos=title_layout.pos)
        title_label = Label(
            text='News Topic Analyzer', 
            font_size='22sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            size_hint_x=0.9
        )
        search_icon = ImageButton(
            source='/storage/emulated/0/NEWSANALYST/search_icon.png',
            size_hint=(None, None),
            size=(40, 40),
            pos_hint={'center_y': 0.5}
        )
        title_layout.add_widget(title_label)
        title_layout.add_widget(search_icon)
        self.add_widget(title_layout)

    def _create_input_field(self):
        self.query_input = TextInput(
            hint_text='Type topic to analyze',
            size_hint_y=None,
            height=80,
            multiline=True,
            padding=[15, 15, 15, 15],
            background_color=(1, 1, 1, 1),
            foreground_color=(0.2, 0.2, 0.2, 1),
            cursor_color=(0.2, 0.6, 0.8, 1),
            font_size='18sp'
        )
        self.add_widget(self.query_input)

    def _create_analyze_button(self):
        button_layout = BoxLayout(size_hint_y=None, height=50, padding=[0, 10, 0, 10])
        self.analyze_button = Button(
            text='Analyze',
            size_hint_x=1,
            background_color=(0.2, 0.6, 0.8, 1),
            color=(1, 1, 1, 1)
        )
        self.analyze_button.bind(on_press=self.analyze_news)
        button_layout.add_widget(self.analyze_button)
        self.add_widget(button_layout)

    def _create_results_area(self):
        results_container = BoxLayout(orientation='vertical', size_hint_y=1)
        with results_container.canvas.before:
            Color(1, 1, 1, 1)  # White background
            Rectangle(size=results_container.size, pos=results_container.pos)
            Color(0.8, 0.8, 0.8, 1)  # Light gray border
            Line(rectangle=(results_container.x, results_container.y, results_container.width, results_container.height))

        self.results_scroll = ScrollView(size_hint=(1, 1))
        self.results_label = Label(
            text='Results will appear here',
            size_hint_y=None,
            color=(0.2, 0.2, 0.2, 1),
            text_size=(Window.width - 60, None),
            halign='left',
            valign='top',
            padding=[10, 10]
        )
        self.results_label.bind(texture_size=self.results_label.setter('size'))
        self.results_scroll.add_widget(self.results_label)
        results_container.add_widget(self.results_scroll)
        self.add_widget(results_container)

    def analyze_news(self, instance):
        query = self.query_input.text
        self.results_label.text = 'Analyzing...'
        self.analyze_callback(query)

    def update_results(self, results):
        self.results_label.text = results