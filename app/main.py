import sys
import menu
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = menu.Menu()
    window.show()
    app.exec()

