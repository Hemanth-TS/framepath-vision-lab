import unittest
from framepath import components, iou, render_frame, Tracker

class VisionTests(unittest.TestCase):
    def test_components(self):
        image=render_frame(20,10,[(1,1,5,5),(10,1,14,5)])
        self.assertEqual(components(image),[(1,1,5,5),(10,1,14,5)])
    def test_noise_removed(self):
        self.assertEqual(components([[255,0],[0,0]]),[])
    def test_diagonal_is_separate(self):
        self.assertEqual(len(components([[255,0],[0,255]],min_area=1)),2)
    def test_iou(self):
        self.assertAlmostEqual(iou((0,0,4,4),(2,0,6,4)),1/3)
        self.assertEqual(iou((0,0,1,1),(2,2,3,3)),0)
    def test_count_once(self):
        t=Tracker(line_x=5)
        for x in [0,1,2,3,4,3,4]: t.update([(x,0,x+4,4)])
        self.assertEqual(t.crossings,1)
        self.assertEqual(len(t.tracks),1)
    def test_occlusion(self):
        t=Tracker(max_missed=1)
        t.update([(0,0,4,4)]); t.update([])
        self.assertIn(1,t.tracks)
        t.update([]); self.assertEqual(t.tracks,{})
    def test_one_to_one(self):
        t=Tracker(); t.update([(0,0,10,10)])
        t.update([(1,0,11,10),(2,0,12,10)])
        self.assertEqual(len(t.tracks),2)
    def test_bad_image(self):
        with self.assertRaises(ValueError): components([[0],[0,0]])
        with self.assertRaises(ValueError): components([[256]])
    def test_bad_box(self):
        with self.assertRaises(ValueError): Tracker().update([(1,1,0,0)])
        with self.assertRaises(ValueError): render_frame(4,4,[(0,0,5,5)])
