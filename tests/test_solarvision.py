import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from solarvision.data import read_labels, split, load, digest
from solarvision.features import extract, batch, occlusion
from solarvision.model import fit, predict, select_threshold, metrics, bootstrap
from solarvision.review import ReviewQueue
from solarvision.report import render

class DataTests(unittest.TestCase):
    def test_split_disjoint_reproducible_and_ambiguous_excluded(self):
        records=[{'label':float(i%2),'type':'mono' if (i//2)%2 else 'poly'} for i in range(100)]
        records += [{'label':.33,'type':'mono'}]
        a,b=split(records),split(records)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        merged=np.concatenate(list(a.values()))
        self.assertEqual(len(set(merged)),101)
        self.assertEqual(a['ambiguous'].tolist(),[100])
        self.assertEqual([len(a[k]) for k in ['train','validation','test']],[60,20,20])
        for role in ['train','validation','test']:
            self.assertEqual(len({(records[i]['label'],records[i]['type']) for i in a[role]}),4)

    def test_invalid_annotations(self):
        for text in ['../bad.png 0 mono','images/a.png nan mono','images/a.png 0 wrong','images/a.png 1.1 mono',
                     'images/a.png 0 mono\nimages/a.png 1 mono']:
            with self.subTest(text=text),tempfile.TemporaryDirectory() as d:
                (Path(d)/'labels.csv').write_text(text)
                with self.assertRaises(ValueError): read_labels(d)

    def test_integrity_and_decoded_duplicate_removal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'images').mkdir()
            a=np.zeros((300,300),dtype=np.uint8)
            Image.fromarray(a).save(p/'images/a.png',compress_level=1)
            Image.fromarray(a).save(p/'images/b.png',compress_level=9)
            (p/'labels.csv').write_text('images/a.png 0 mono\nimages/b.png 0 mono')
            manifest={'count':2,'labels_sha256':digest(p/'labels.csv'),
                      'images':{x:digest(p/x) for x in ['images/a.png','images/b.png']}}
            (p/'manifest.json').write_text(json.dumps(manifest))
            records,images,_,duplicates=load(p)
            self.assertEqual(len(records),1);self.assertEqual(duplicates,['images/b.png'])
            (p/'images/a.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Image checksum'):load(p)

    def test_changed_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'labels.csv').write_text('images/a.png 0 mono')
            (p/'manifest.json').write_text(json.dumps({'labels_sha256':'bad'}))
            with self.assertRaisesRegex(ValueError,'Labels checksum'):load(p)

class FeatureTests(unittest.TestCase):
    def test_constant_image_finite_and_correct_dimensions(self):
        x=extract(np.zeros((300,300),dtype=np.uint8))
        self.assertEqual(x.shape,(1032,));self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[0],0);self.assertEqual(x[1],0)

    def test_texture_changes_for_same_mean(self):
        a=np.tile(np.array([0,255],dtype=np.uint8), (96,48))
        b=np.concatenate([np.zeros((48,96),dtype=np.uint8),np.full((48,96),255,dtype=np.uint8)])
        x,y=extract(a),extract(b)
        self.assertAlmostEqual(x[0],y[0]);self.assertFalse(np.allclose(x[42:],y[42:]))

    def test_invalid_images(self):
        for image in [np.zeros((96,96,3)),np.zeros((4,4)),np.full((96,96),np.nan),np.full((96,96),256)]:
            with self.subTest(shape=image.shape),self.assertRaises(ValueError):extract(image)

    def test_deterministic_and_no_mutation(self):
        a=np.random.default_rng(1).integers(0,256,(96,96),dtype=np.uint8);before=a.copy()
        np.testing.assert_array_equal(extract(a),extract(a));np.testing.assert_array_equal(a,before)

    def test_occlusion_constant_predictor_zero(self):
        a=np.full((96,96),120,dtype=np.uint8)
        result=occlusion(a,lambda X:np.full(len(X),.4))
        np.testing.assert_array_equal(result,np.zeros((6,6)))

class ModelTests(unittest.TestCase):
    def test_threshold_meets_recall_with_inclusive_ties(self):
        y=[1,1,0,0];s=[.9,.6,.6,.1]
        t,m=select_threshold(y,s,target=1)
        self.assertEqual(t,.6);self.assertEqual(m['recall'],1);self.assertAlmostEqual(m['precision'],2/3)

    def test_invalid_threshold_inputs(self):
        for y,s,target in [([0,0],[.1,.2],.9),([0,1],[.1,np.nan],.9),([0,1],[.1,.2],0)]:
            with self.assertRaises(ValueError):select_threshold(y,s,target)

    def test_metrics_confusion_and_empty_flags(self):
        m=metrics([0,0,1,1],[.1,.8,.2,.9],.5)
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertEqual(m['recall'],.5)
        self.assertEqual(metrics([0,1],[.1,.2],1)['precision'],0)

    def test_exported_inference_matches_sklearn(self):
        rng=np.random.default_rng(5);X=rng.normal(size=(40,8));y=np.arange(40)%2
        scaler=StandardScaler().fit(X);clf=LogisticRegression().fit(scaler.transform(X),y)
        model={'feature_version':1,'width':8,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
               'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0])}
        np.testing.assert_allclose(predict(json.loads(json.dumps(model)),X),clf.predict_proba(scaler.transform(X))[:,1])

    def test_test_data_cannot_change_fit_or_threshold(self):
        rng=np.random.default_rng(6);X=rng.normal(size=(100,50));y=np.arange(100)%2
        parts={'train':np.arange(60),'validation':np.arange(60,80),'test':np.arange(80,100)}
        first,_=fit(X,y,parts);X[80:]=10000;y[80:]=1-y[80:]
        second,_=fit(X,y,parts)
        self.assertEqual(first,second)

    def test_bootstrap_reproducible_and_bounded(self):
        a=bootstrap([0,0,1,1],[.1,.2,.8,.9],.5,30)
        self.assertEqual(a,bootstrap([0,0,1,1],[.1,.2,.8,.9],.5,30))
        self.assertEqual(a['recall'],[1.,1.])

    def test_bad_model_and_nonfinite_features_rejected(self):
        with self.assertRaises(ValueError):predict({'feature_version':2},np.ones((1,3)))
        with self.assertRaises(ValueError):predict({'feature_version':1,'width':3},np.full((1,3),np.nan))

class ReviewTests(unittest.TestCase):
    def test_lifecycle_persistence_and_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.sqlite';q=ReviewQueue(p)
            q.add('a',.8,.5);self.assertEqual(q.cases()[0]['status'],'QUEUED')
            q.review('a','uncertain','Needs a second EL exposure');q.close()
            q=ReviewQueue(p);self.assertEqual(q.cases()[0]['decision'],'uncertain')
            self.assertEqual(q.db.execute('SELECT count(*) FROM events').fetchone()[0],2);q.close()

    def test_low_score_is_still_queued(self):
        q=ReviewQueue(':memory:');q.add('a',.1,.5)
        self.assertEqual(q.cases()[0]['priority'],'LOW_SCORE');self.assertEqual(q.cases()[0]['status'],'QUEUED');q.close()

    def test_duplicate_does_not_add_event(self):
        q=ReviewQueue(':memory:');q.add('a',.5,.5)
        with self.assertRaises(sqlite3.IntegrityError):q.add('a',.8,.5)
        self.assertEqual(q.db.execute('SELECT count(*) FROM events').fetchone()[0],1);q.close()

    def test_missing_or_repeated_review(self):
        q=ReviewQueue(':memory:')
        with self.assertRaises(ValueError):q.review('missing','clear','checked')
        q.add('a',.8,.5);q.review('a','clear','checked')
        with self.assertRaises(ValueError):q.review('a','defect','change')
        q.close()

    def test_invalid_case_and_decision(self):
        q=ReviewQueue(':memory:')
        for score in [-1,2,float('nan')]:
            with self.assertRaises(ValueError):q.add('a',score,.5)
        q.add('a',.5,.5)
        with self.assertRaises(ValueError):q.review('a','unknown','note')
        with self.assertRaises(ValueError):q.review('a','defect',' ')
        self.assertEqual(q.cases()[0]['status'],'QUEUED');q.close()

class ReportTests(unittest.TestCase):
    def test_json_script_escape(self):
        page=render({'text':'</script><script>alert(1)</script>'})
        self.assertNotIn('</script><script>alert',page)
        self.assertIn('\\u003c/script\\u003e',page)

if __name__=='__main__':unittest.main()
