import tkinter as tk
root = tk.Tk()

width_px = root.winfo_screenwidth()
height_px = root.winfo_screenheight()
width_mm = root.winfo_screenmmwidth()
height_mm = root.winfo_screenmmheight()
# 2.54 cm = in
width_in = width_mm / 25.4
height_in = height_mm / 25.4
width_dpi = width_px/width_in
height_dpi = height_px/height_in

print('Width: %i px, Height: %i px' % (width_px, height_px))
print('Width: %i mm, Height: %i mm' % (width_mm, height_mm))
print('Width: %f in, Height: %f in' % (width_in, height_in))
print('Width: %f dpi, Height: %f dpi' % (width_dpi, height_dpi))

import ctypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print('Size is %f %f' % (w, h))

curr_dpi = w*96/width_px
print('Current DPI is %f' % (curr_dpi))