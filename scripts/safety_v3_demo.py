"""Small reproducible safety-v3 logic demonstration. No diagnosis and no images.

Run from project root: python -m scripts.safety_v3_demo
"""
from radreport.safety import report_claims, self_consistency_details, entity_consistency, route


def main():
    samples = [
        'No acute pulmonary disease.',
        'No acute cardiopulmonary abnormality.',
        'No acute cardiopulmonary abnormality.',
        'No active disease.',
        'No acute cardiopulmonary findings.',
    ]
    print('Equivalent no-acute reports:', self_consistency_details(samples))
    print('Normal draft with simulated high image-model pneumonia score:',
          entity_consistency('Normal chest.',{'Pneumonia':0.91}))
    print('Named pneumonia with simulated low image-model score:',
          entity_consistency('Right lower lobe pneumonia.', {'Pneumonia':0.08}))
    print('IMPORTANT: synthetic scores are only a logic test; no X-ray analysis occurred.')


if __name__=='__main__':
    main()
