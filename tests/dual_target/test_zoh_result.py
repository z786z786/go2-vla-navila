import unittest
from src.dual_target.zoh_result_contract import normalized_result


class ResultTests(unittest.TestCase):
    def test_full_legacy_timeout_is_derived_without_mutation(self):
        original={'status':'FAILED_TIMEOUT'}
        result=normalized_result(original,1500,1500)
        self.assertEqual(result['steps'],1500)
        self.assertNotIn('steps',original)
        self.assertIn('steps_provenance',result)

    def test_partial_or_missing_success_not_fabricated(self):
        for result,a,b in [({'status':'FAILED_TIMEOUT'},1499,1499),
            ({'status':'FAILED_TIMEOUT'},1500,1499),({'status':'success'},1500,1500),
            ({'status':'FAILED_COLLISION'},50,50)]:
            with self.assertRaises(ValueError):normalized_result(result,a,b)

    def test_explicit_steps_checked(self):
        self.assertEqual(normalized_result({'status':'success','steps':50},50,50)['steps'],50)
        for value in [49,51,True,50.0]:
            with self.assertRaises(ValueError):normalized_result({'steps':value},50,50)


if __name__=='__main__':unittest.main()
