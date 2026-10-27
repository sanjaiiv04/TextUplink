import unittest
from types import SimpleNamespace
import numpy as np
from textuplink.transcript import annotate_context, colored_html, review_annotations, transcribe_words


def word(text,start,end):
    return dict(word=text,start_s=start,end_s=end,emotion='Unlabelled',annotation_source='None',model_score=None)


class TranscriptTests(unittest.TestCase):
    def test_qwen_alignment_is_used_without_fabricated_word_times(self):
        result=SimpleNamespace(language='English',text='Help me',time_stamps=[
            SimpleNamespace(text='Help',start_time=.1,end_time=.4),
            SimpleNamespace(text='me',start_time=.5,end_time=.8)])
        class Model:
            def transcribe(self,**kwargs):
                assert kwargs['return_time_stamps'] is True
                assert kwargs['audio'][1]==16000
                return [result]
        rows,language=transcribe_words(np.zeros(24000,dtype=np.float32),Model())
        self.assertEqual(language,'English')
        self.assertEqual([(r['word'],r['start_s'],r['end_s']) for r in rows],[('Help',.1,.4),('me',.5,.8)])
        result.time_stamps=None
        with self.assertRaises(ValueError):transcribe_words(np.zeros(24000,dtype=np.float32),Model())

    def test_context_labels_are_shared_and_uncertain_predictions_abstain(self):
        class Model:
            def __init__(self): self.calls=0
            def estimate(self,audio):
                self.calls+=1
                return ('Angry',.9) if self.calls==1 else ('Happy',.4)
        model=Model()
        rows=[word('Help',0,.5),word('me',1,1.5),word('please',4,4.5)]
        result=annotate_context(rows,np.zeros(6*24000),model,.6)
        self.assertEqual(model.calls,2)
        self.assertEqual([r['emotion'] for r in result],['Angry','Angry','Uncertain'])
        self.assertEqual(result[0]['context_start_s'],result[1]['context_start_s'])
        self.assertEqual(result[2]['model_score'],.4)
        self.assertEqual(rows[0]['emotion'],'Unlabelled')

    def test_colored_transcript_escapes_text_and_attributes(self):
        row=word('<script>alert(1)</script>',0,1)
        row['annotation_source']='" onmouseover="evil'
        output=colored_html([row])
        self.assertNotIn('<script>',output)
        self.assertIn('&lt;script&gt;',output)
        self.assertIn('&quot;',output)
        self.assertNotIn('title="Unlabelled · " onmouseover',output)

    def test_human_override_does_not_become_model_prediction(self):
        original=dict(word('help',0,1),emotion='Angry',annotation_source='Phrase-context model estimate',model_score=.7)
        edited=dict(original,emotion='Sad')
        result=review_annotations([edited],[original])[0]
        self.assertEqual(result['annotation_source'],'Human override')
        self.assertEqual(result['model_score'],.7)
        self.assertEqual(original['emotion'],'Angry')


if __name__=='__main__':unittest.main()
