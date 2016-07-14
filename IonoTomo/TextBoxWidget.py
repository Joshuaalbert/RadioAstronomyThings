
# coding: utf-8

# In[1]:

from matplotlib.widgets import AxesWidget
import numpy as np
from matplotlib import rcParams
from matplotlib import six
#import itertools.izip as zip

class TextBox(AxesWidget):
    """
    A GUI neutral text input box.
    For the text box to remain responsive you must keep a reference to it.
    The following attributes are accessible:
      *ax*
        The :class:`matplotlib.axes.Axes` the button renders into.
      *label*
        A :class:`matplotlib.text.Text` instance.
      *color*
        The color of the text box when not hovering.
      *hovercolor*
        The color of the text box when hovering.
    Call :meth:`on_text_change` to be updated whenever the text changes.
    Call :meth:`on_submit` to be updated whenever the user hits enter or
    leaves the text entry field.
    """

    def __init__(self, ax, label, initial='',
                 color='.95', hovercolor='1', label_pad=.01):
        """
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The :class:`matplotlib.axes.Axes` instance the button
            will be placed into.
        label : str
            Label for this text box. Accepts string.
        initial : str
            Initial value in the text box
        color : color
            The color of the box
        hovercolor : color
            The color of the box when the mouse is over it
            
        label_pad : float
            the distance between the label and the right side of the textbox
        """
        AxesWidget.__init__(self, ax)

        self.DIST_FROM_LEFT = .05

        self.params_to_disable = []
        for key in rcParams.keys():
            if u'keymap' in key:
                self.params_to_disable += [key]

        self.text = initial
        self.label = ax.text(-label_pad, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='right',
                             transform=ax.transAxes)
        self.text_disp = self._make_text_disp(self.text)

        self.cnt = 0
        self.change_observers = {}
        self.submit_observers = {}

        # If these lines are removed, the cursor won't appear the first
        # time the box is clicked:
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        self.cursor_index = 0

        # Because this is initialized, _render_cursor
        # can assume that cursor exists.
        self.cursor = self.ax.vlines(0, 0, 0)
        self.cursor.set_visible(False)

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        self.connect_event('key_press_event', self._keypress)
        self.connect_event('resize_event', self._resize)
        ax.set_navigate(False)
        ax.set_axis_bgcolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.color = color
        self.hovercolor = hovercolor

        self._lastcolor = color

        self.capturekeystrokes = False

    def _make_text_disp(self, string):
        return self.ax.text(self.DIST_FROM_LEFT, 0.5, string,
                            verticalalignment='center',
                            horizontalalignment='left',
                            transform=self.ax.transAxes)

    def _rendercursor(self):
        # this is a hack to figure out where the cursor should go.
        # we draw the text up to where the cursor should go, measure
        # and save its dimensions, draw the real text, then put the cursor
        # at the saved dimensions

        widthtext = self.text[:self.cursor_index]
        no_text = False
        if(widthtext == "" or widthtext == " " or widthtext == "  "):
            no_text = widthtext == ""
            widthtext = ","

        wt_disp = self._make_text_disp(widthtext)

        self.ax.figure.canvas.draw()
        bb = wt_disp.get_window_extent()
        inv = self.ax.transData.inverted()
        bb = inv.transform(bb)
        wt_disp.set_visible(False)
        if no_text:
            bb[1, 0] = bb[0, 0]
        # hack done
        self.cursor.set_visible(False)

        self.cursor = self.ax.vlines(bb[1, 0], bb[0, 1], bb[1, 1])
        self.ax.figure.canvas.draw()

    def _notify_submit_observers(self):
        for cid, func in six.iteritems(self.submit_observers):
                func(self.text)

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)

    def _keypress(self, event):
        if self.ignore(event):
            return
        if self.capturekeystrokes:
            key = event.key

            if(len(key) == 1):
                self.text = (self.text[:self.cursor_index] + key +
                             self.text[self.cursor_index:])
                self.cursor_index += 1
            elif key == "right":
                if self.cursor_index != len(self.text):
                    self.cursor_index += 1
            elif key == "left":
                if self.cursor_index != 0:
                    self.cursor_index -= 1
            elif key == "home":
                self.cursor_index = 0
            elif key == "end":
                self.cursor_index = len(self.text)
            elif(key == "backspace"):
                if self.cursor_index != 0:
                    self.text = (self.text[:self.cursor_index - 1] +
                                 self.text[self.cursor_index:])
                    self.cursor_index -= 1
            elif(key == "delete"):
                if self.cursor_index != len(self.text):
                    self.text = (self.text[:self.cursor_index] +
                                 self.text[self.cursor_index + 1:])

            self.text_disp.remove()
            self.text_disp = self._make_text_disp(self.text)
            self._rendercursor()
            for cid, func in six.iteritems(self.change_observers):
                func(self.text)
            if key == "enter":
                self._notify_submit_observers()

    def begin_typing(self, x):
        self.capturekeystrokes = True
        #disable command keys so that the user can type without
        #command keys causing figure to be saved, etc
        self.reset_params = {}
        for key in self.params_to_disable:
            self.reset_params[key] = rcParams[key]
            rcParams[key] = []
        #now, we have to figure out where the cursor goes.
        #approximate it based on assuming all characters the same length
        self.cursor_index = len(self.text)
        self._rendercursor()

    def stop_typing(self):
        notifysubmit = False
        # because _notify_submit_users might throw an error in the
        # user's code, we only want to call it once we've already done
        # our cleanup.
        if self.capturekeystrokes:
            #since the user is no longer typing, 
            #reactivate the standard command keys
            for key in self.params_to_disable:
                rcParams[key] = self.reset_params[key]
            notifysubmit = True
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self.ax.figure.canvas.draw()
        if notifysubmit:
            self._notify_submit_observers()


    def _click(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            self.stop_typing()
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        if not(self.capturekeystrokes):
            self.begin_typing(event.x)

    def _resize(self, event):
        self.stop_typing()

    def _motion(self, event):
        if self.ignore(event):
            return
        if event.inaxes == self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            if self.drawon:
                self.ax.figure.canvas.draw()

    def on_text_change(self, func):
        """
        When the text changes, call this *func* with event.
        A connection id is returned which can be used to disconnect.
        """
        cid = self.cnt
        self.change_observers[cid] = func
        self.cnt += 1
        return cid

    def on_submit(self, func):
        """
        When the user hits enter or leaves the submision box, call this
        *func* with event.
        A connection id is returned which can be used to disconnect.
        """
        cid = self.cnt
        self.submit_observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass


# In[ ]:



