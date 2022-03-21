import numpy as np
import sys

files = np.loadtxt(sys.argv[1], dtype=str, unpack=True)

count = 0

print("<!DOCTYPE html>\n<html>\n<head>\n<style>\n* {\n  box-sizing: border-box;\n}\n\n.column {\n  float: left;\n  width: 50.00%;\n  padding: 5px;\n}\n\n/* Clearfix (clear floats) */\n.row::after {\n  content: \"\";\n  clear: both;\n  display: table;\n}\n</style>\n</head>\n<body>\n\n<h2>Gawa results</h2>\n<p>Including all objects</p>\n\n<div class=\"row\">\n")

for i in files:

    if count == 1:
        print("  <div class=\"column\">\n    <img src=\"" + i + "\" alt=\"\" style=\"width:100%\">\n  </div>\n\n</div>\n\n<div class=\"row\">\n")
        count = 0
    else:
        print("  <div class=\"column\">\n    <img src=\"" + i + "\" alt=\"\" style=\"width:100%\">\n  </div>\n")
        count += 1
        
print("</div>\n\n</body>\n</html>")
