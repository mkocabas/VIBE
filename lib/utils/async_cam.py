import cv2
from threading import Thread
import time
import numpy as np

# Install winCurses from pypi for only windows
import curses


class AsyncCamera(object):
	def __init__(self,cam,display = False):

		self.prev_frame_lis = []
		self.frame = None
		self.bbox = None
		self.display_image = None
		self.capture = cv2.VideoCapture(cam)
		self.stop = False
		self.display = display

		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		if self.display:
			self.start_displaying()


	def check_key_press(self):
		def key_press_wrapper(stdscr):
			stdscr.nodelay(True)  
			return stdscr.getch()==ord('q')
		return curses.wrapper(key_press_wrapper)

	def stop_cam(self):
		self.capture.release()
		self.stop = True


	def show_frame(self):
		image = np.copy(self.display_image)
		if self.status and self.display_image is not None:
			c_x,c_y,w,h = self.bbox
			start_point = (int(c_x-w/2),int(c_y-h/2))
			end_point = (int(c_x+w/2),int(c_y+h/2))
			cv2.rectangle(image, start_point, end_point, (0, 0, 255) , 2) 
			cv2.imshow('bbox', image)

		key = cv2.waitKey(1)
		if key == ord('q'):
			self.stop_cam()
			cv2.destroyAllWindows()



	def update(self):
		while True and not self.stop:
			if self.capture.isOpened():
				if(not self.display):
					if(self.check_key_press()):
						self.stop_cam()
				(self.status, self.frame) = self.capture.read()
				self.prev_frame_lis.append(self.frame)



	def read(self):
		if(self.frame is not None):
			return True,self.prev_frame_lis
		else:
			return False,None

	def del_frame_lis(self):
		self.prev_frame_lis = []
		self.status = False


	def start_displaying(self):
		def start_display_thread():
			while True:
				try:
					self.show_frame()
				except AttributeError:
					pass
		self.displayThread= Thread(target=start_display_thread, args=())
		self.displayThread.daemon = True
		self.displayThread.start()

	def set_bounding_box(self,bbox):
		self.bbox = bbox

	def set_display_image(self,display_image):
		self.display_image = display_image