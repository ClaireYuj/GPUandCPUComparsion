#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   konan_yu@163.com
@Author  :   Yu
@Date    :   2023/3/16 1:16
------------      --------    -----------

"""
import matplotlib.animation as animation


class AnimationDraw(animation.TimedAnimation):
    """
    Rewrite Animation using a fixed set of `.Artist` objects.

    Before creating an instance, all plotting should have taken place
    and the relevant artists saved.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    artists : list
        Each list entry is a collection of `.Artist` objects that are made
        visible on the corresponding frame.  Other artists are made invisible.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """

    def __init__(self, fig, artists, *args, **kwargs):
        # Internal list of artists drawn in the most recent frame.
        self._drawn_artists = [[]]

        # Use the list of artists as the framedata, which will be iterated
        # over by the machinery.
        self._framedata = artists
        super().__init__(fig, *args, **kwargs)

    def _init_draw(self):
        super()._init_draw()
        # Make all the artists involved in *any* frame invisible
        figs = set()
        for f in self.new_frame_seq():
            for artist in f:

                for artist_i in artist:
                    artist_i.set_visible(False)

                    # artist.set_visible(False)
                    artist_i.set_animated(self._blit)

                # Assemble a list of unique figures that need flushing
                    if artist_i.get_figure() not in figs:
                        figs.add(artist_i.get_figure())

        # Flush the needed figures
        for fig in figs:
            fig.canvas.draw_idle()

    def _pre_draw(self, framedata, blit):
        """Clears artists from the last frame."""
        if blit:
            # Let blit handle clearing
            self._blit_clear(self._drawn_artists)
        else:
            # Otherwise, make all the artists from the previous frame invisible
            for artist in self._drawn_artists:
                for artist_a in artist:
                    artist_a.set_visible(False)


    def _draw_frame(self, artists):
        # Save the artists that were passed in as framedata for the other
        # steps (esp. blitting) to use.
        self._drawn_artists = artists

        # Make all the artists from the current frame visible
        for artist in artists:
            for artist_i in artist:
                artist_i.set_visible(True)

