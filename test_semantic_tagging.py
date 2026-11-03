"""
Name             : Aman Banka
Registration No. : 261100660030
Course           : AME 5151 - Applied Linear Algebra Lab
Assignment       : Assignment 2 - Semantic Tagging with GloVe and NumPy
Institution      : Manipal School of Information Sciences, MAHE, Manipal
Faculty          : Dr. Devi Prasad
"""

"""
Unit tests for semantic_tagging.py (AME 5151 Assignment 2).

Run with either:
    python -m pytest -v test_semantic_tagging.py
    python -m unittest -v test_semantic_tagging
"""

import unittest

import numpy as np

from vector import Vector
import semantic_tagging as st


class TestSemanticTagging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = st.load_model()
        cls.tag_names, cls.T = st.build_tag_matrix(cls.model, st.TAGS)
        # A small text with punctuation, a hashtag, a number, repetition,
        # stopwords and two made-up OOV words.
        cls.text = ("Our #ResearchExcellence: research at the university, "
                    "research & innovation in 2026! Students build qwzxtech "
                    "and blorptastic labs.")
        _, cls.tokens = st.preprocess(cls.text)
        cls.in_vocab, cls.oov, cls.W = st.build_text_matrix(cls.model, cls.tokens)
        cls.S_vector = np.array(st.similarity_matrix_vector_class(cls.W, cls.T))
        cls.S_numpy = st.similarity_matrix_numpy(cls.W, cls.T)

    # 1. T.shape == (20, 50)
    def test_01_tag_matrix_shape(self):
        self.assertEqual(self.T.shape, (20, 50))
        self.assertEqual(len(self.tag_names), 20)

    # 1b. single-word row equals model[tag]; multi-word row is the mean
    def test_01b_tag_rows(self):
        j = self.tag_names.index("research")
        np.testing.assert_allclose(self.T[j], self.model["research"])
        j = self.tag_names.index("higher education")
        mean = (self.model["higher"].astype(np.float64)
                + self.model["education"].astype(np.float64)) / 2
        np.testing.assert_allclose(self.T[j], mean)

    # 2. W.shape == (n, 50) with n in-vocabulary tokens (repeats kept)
    def test_02_text_matrix_shape(self):
        n = len(self.in_vocab)
        self.assertGreater(n, 0)
        self.assertEqual(self.W.shape, (n, 50))
        self.assertEqual(self.in_vocab.count("research"), 2)   # repetition kept
        for i, tok in enumerate(self.in_vocab):
            np.testing.assert_allclose(self.W[i], self.model[tok])

    # 3. S.shape == (n, 20)
    def test_03_similarity_shape(self):
        n = len(self.in_vocab)
        self.assertEqual(self.S_numpy.shape, (n, 20))
        self.assertEqual(self.S_vector.shape, (n, 20))

    # 4. one word/tag pair: Vector class agrees with the NumPy entry
    def test_04_single_pair_agrees(self):
        i = self.in_vocab.index("university")
        j = self.tag_names.index("campus")
        expected = Vector(self.model["university"].tolist()).cosine_similarity(
            Vector(self.model["campus"].tolist()))
        self.assertAlmostEqual(self.S_numpy[i, j], expected, places=6)
        # an identical word/tag pair gives exactly cosine 1
        i = self.in_vocab.index("research")
        j = self.tag_names.index("research")
        self.assertAlmostEqual(self.S_numpy[i, j], 1.0, places=12)

    # 5. np.allclose(S_vector, S_numpy) is True
    def test_05_allclose(self):
        self.assertTrue(np.allclose(self.S_vector, self.S_numpy,
                                    rtol=st.RTOL, atol=st.ATOL))
        ok, max_diff = st.verify(self.S_vector, self.S_numpy)
        self.assertTrue(ok)
        self.assertLess(max_diff, 1e-12)

    # 6. OOV token excluded from W and reported as OOV
    def test_06_oov_excluded_and_reported(self):
        for word in ("qwzxtech", "blorptastic", "researchexcellence"):
            self.assertIn(word, self.tokens)
            self.assertIn(word, self.oov)
            self.assertNotIn(word, self.in_vocab)
        self.assertEqual(self.W.shape[0], len(self.tokens) - len(self.oov))

    # 7. a tag with an OOV component raises an error (not silently dropped)
    def test_07_oov_tag_raises(self):
        bad_single = st.TAGS[:19] + ["qwzxtech"]
        with self.assertRaises(ValueError):
            st.build_tag_matrix(self.model, bad_single)
        bad_multi = st.TAGS[:19] + ["research qwzxtech"]
        with self.assertRaises(ValueError):
            st.build_tag_matrix(self.model, bad_multi)

    # extra: preprocessing order and filters
    def test_08_preprocessing(self):
        raw, toks = st.preprocess("The AI-first Lab, at MAHE: we did it!! #Research2026")
        self.assertEqual(raw, 9)
        self.assertEqual(toks, ["first", "lab", "mahe", "research2026"])

    # extra: n = 0 is detected (no usable words)
    def test_09_empty_text_matrix(self):
        _, toks = st.preprocess("we are at the #qwzxtech!")
        in_vocab, oov, W = st.build_text_matrix(self.model, toks)
        self.assertEqual(W.shape, (0, 50))
        self.assertEqual(in_vocab, [])

    # extra: max pooling picks the column maximum and its word
    def test_10_rank_tags(self):
        ranking = st.rank_tags(self.tag_names, self.S_numpy, self.in_vocab)
        self.assertEqual(len(ranking), 20)
        scores = [r["score"] for r in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for r in ranking:
            j = self.tag_names.index(r["tag"])
            self.assertAlmostEqual(r["score"], self.S_numpy[:, j].max())
            i = self.in_vocab.index(r["word"])
            self.assertAlmostEqual(self.S_numpy[i, j], r["score"])

    # extra: zero rows are rejected before division
    def test_11_zero_row_rejected(self):
        with self.assertRaises(ValueError):
            st.similarity_matrix_numpy(np.zeros((1, 50)), self.T)


class TestVectorClass(unittest.TestCase):
    def test_vector_ops(self):
        u, v = Vector((3.0, 4.0)), Vector((4.0, 3.0))
        self.assertEqual(u.add(v), Vector((7.0, 7.0)))
        self.assertEqual(u.scale(2), Vector((6.0, 8.0)))
        self.assertEqual(u.dot(v), 24.0)
        self.assertEqual(u.norm(), 5.0)
        self.assertAlmostEqual(u.cosine_similarity(v), 24 / 25)
        with self.assertRaises(ValueError):
            u.cosine_similarity(Vector((0.0, 0.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
