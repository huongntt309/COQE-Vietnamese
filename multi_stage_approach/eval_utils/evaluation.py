import json
import os
import pandas as pd

ALL_ELEMENTS = ['subject', 'object', 'aspect', 'predicate']
ALL_LABELS = ['EQL', 'DIF', 'COM', 'COM+', 'COM-', 'SUP', 'SUP+', 'SUP-']


def evaluate_intra(gold_dir, pred_dir):
    def calculate_prf(ret, is_tuple=False):
        # Calculate precision, recall, and F1 for each matching criterion
        # only calculate proportional match for NER, skip P if calculating for tuple
        for m in ['E', 'B'] if is_tuple else ['E', 'P', 'B']:
            ret[f'{m}-P'] = ret[f'{m}-TP'] / (ret[f'{m}-TP'] + ret[f'{m}-FP']) if (ret[f'{m}-TP'] + ret[
                f'{m}-FP']) > 0 else 0.0
            ret[f'{m}-R'] = ret[f'{m}-TP'] / (ret[f'{m}-TP'] + ret[f'{m}-FN']) if (ret[f'{m}-TP'] + ret[
                f'{m}-FN']) > 0 else 0.0
            ret[f'{m}-F1'] = 2 * (ret[f'{m}-P'] * ret[f'{m}-R']) / (ret[f'{m}-P'] + ret[f'{m}-R']) if (ret[f'{m}-P'] +
                                                                                                       ret[
                                                                                                           f'{m}-R']) > 0 else 0.0

        return ret

    def get_and_add_infix(ret, infix):
        return {
            key.replace('-', f'-{infix}-'): ret[key] for key in ret
            if key.endswith('-P') or key.endswith('-R') or key.endswith('-F1')
        }

    def calculate_ner_metrics(sent_gold_tuples, sent_pred_tuples):
        def calculate_sentence_ner_metrics(gold_entities, pred_entities):
            def entity_to_set(entity):
                return set(int(k.split('&&')[0]) for k in entity)

            def exact_match(entity_set1, entity_set2):
                return int(entity_set1 == entity_set2)

            def proportional_match(entity_set1, entity_set2):
                intersection = entity_set1.intersection(entity_set2)
                return len(intersection) / len(entity_set1)

            def binary_match(entity_set1, entity_set2):
                return int(len(entity_set1.intersection(entity_set2)) != 0)

            try:

                preds = [entity_to_set(e) for e in pred_entities]
                golds = [entity_to_set(e) for e in gold_entities]
            except:
                print()

            ret = {
                'E-TP': 0, 'E-FP': 0, 'E-FN': 0,  # exact match
                'P-TP': 0, 'P-FP': 0, 'P-FN': 0,  # proportional match
                'B-TP': 0, 'B-FP': 0, 'B-FN': 0,  # binary match
            }

            # calculate TP and FP
            for pred in preds:
                # exact match any entity in gold_entities
                if any(exact_match(pred, k) == 1 for k in golds):
                    ret['E-TP'] += 1
                else:
                    ret['E-FP'] += 1

                # proportional match with the longest matched entity in gold_entities
                max_match = max(proportional_match(pred, k) for k in golds) if len(golds) else 0
                ret['P-TP'] += max_match
                ret['P-FP'] += (1 - max_match)

                # binary match any entity in gold_entities
                if any(binary_match(pred, k) == 1 for k in golds):
                    ret['B-TP'] += 1
                else:
                    ret['B-FP'] += 1

            # calculate FN
            for gold in golds:
                # NOT exact match any entity in pred_entities
                if not any(exact_match(gold, k) == 1 for k in preds):
                    ret['E-FN'] += 1

                # proportional match FN is calculated on the the longest matched entity in pred_entities
                max_match = max(proportional_match(gold, k) for k in preds) if len(preds) else 0
                ret['P-FN'] += (1 - max_match)

                # NOT binary match any entity in pred_entities
                if not any(binary_match(gold, k) == 1 for k in preds):
                    ret['B-FN'] += 1

            return calculate_prf(ret)

        def calculate_ner_metrics_one_element(sent_gold_entities, sent_pred_entities):
            sentence_ner_metrics = [calculate_sentence_ner_metrics(g, p) for g, p in
                                    zip(sent_gold_entities, sent_pred_entities)]

            # calculate TP, FP, FN for all sentences
            ret = {
                k: sum(s_metrics[k] for s_metrics in sentence_ner_metrics)
                for k in [
                    'E-TP', 'E-FP', 'E-FN',  # exact match
                    'P-TP', 'P-FP', 'P-FN',  # proportional match
                    'B-TP', 'B-FP', 'B-FN',  # binary match
                ]
            }

            return calculate_prf(ret)

        # get score for each element type S-subject~index 0, O-object~index 1, A-aspect~index 2, P-predicate~index 3
        element_results = [
            calculate_ner_metrics_one_element(
                [set(tuple(t[k]) for t in gold_tuples if len(t[k]) != 0) for gold_tuples in sent_gold_tuples],
                [set(tuple(t[k]) for t in pred_tuples if len(t[k]) != 0) for pred_tuples in sent_pred_tuples]
            ) for k in ALL_ELEMENTS
        ]

        ret = {
            **get_and_add_infix(element_results[0], 'NER-S'),
            **get_and_add_infix(element_results[1], 'NER-O'),
            **get_and_add_infix(element_results[2], 'NER-A'),
            **get_and_add_infix(element_results[3], 'NER-P'),
        }

        # calculate micro average P, R, F1
        ret.update(
            get_and_add_infix(
                calculate_prf({
                    k: sum(e_metrics[k] for e_metrics in element_results)
                    for k in [
                        'E-TP', 'E-FP', 'E-FN',  # exact match
                        'P-TP', 'P-FP', 'P-FN',  # proportional match
                        'B-TP', 'B-FP', 'B-FN',  # binary match
                    ]
                }),
                'NER-MICRO'
            )
        )

        # calculate macro average P, R, F1
        ret.update(
            get_and_add_infix(
                {
                    k: sum(e_metrics[k] for e_metrics in element_results) / 4
                    for k in [
                    'E-P', 'E-R', 'E-F1',  # exact match
                    'P-P', 'P-R', 'P-F1',  # proportional match
                    'B-P', 'B-R', 'B-F1',  # binary match
                ]
                },
                'NER-MACRO'
            )
        )

        return ret

    def calculate_tuple_metrics(sent_gold_tuples, sent_pred_tuples):
        def calculate_sentence_tuple_metrics(gold_tuples, pred_tuples, omit_label=True):
            def entity_to_set(entity):
                return set(int(k.split('&&')[0]) for k in entity)

            def tuple_to_set(tup):
                return {
                    **{
                        k: entity_to_set(tup[k]) for k in ALL_ELEMENTS
                    }, 'label': tup['label']
                }

            # all element should be exactly matched
            def exact_match(tup_set1, tup_set2):
                return int(
                    all(
                        tup_set1[k] == tup_set2[k]
                        for k in ALL_ELEMENTS
                    )
                    and (omit_label or tup_set1['label'] == tup_set2['label'])
                )

            # all element should be matched at least one token
            def binary_match(tup_set1, tup_set2):
                return int(
                    all(
                        len(tup_set1[k]) == len(tup_set2[k]) == 0 or len(tup_set1[k].intersection(tup_set2[k])) != 0
                        for k in ALL_ELEMENTS
                    )
                    and (omit_label or tup_set1['label'] == tup_set2['label'])
                )

                return int(len(entity_set1.intersection(entity_set2)) != 0)

            preds = [tuple_to_set(t) for t in pred_tuples]
            golds = [tuple_to_set(t) for t in gold_tuples]

            ret = {
                'E-TP': 0, 'E-FP': 0, 'E-FN': 0,  # exact match
                'B-TP': 0, 'B-FP': 0, 'B-FN': 0,  # binary match
            }

            # calculate TP and FP
            for pred in preds:
                # exact match any entity in gold_entities
                if any(exact_match(pred, k) == 1 for k in golds):
                    ret['E-TP'] += 1
                else:
                    ret['E-FP'] += 1

                # binary match any entity in gold_entities
                if any(binary_match(pred, k) == 1 for k in golds):
                    ret['B-TP'] += 1
                else:
                    ret['B-FP'] += 1

            # calculate FN
            for gold in golds:
                # NOT exact match any tuple in pred_tuples
                if not any(exact_match(gold, k) == 1 for k in preds):
                    ret['E-FN'] += 1

                # NOT binary match any tuple in pred_tuples
                if not any(binary_match(gold, k) == 1 for k in preds):
                    ret['B-FN'] += 1

            return calculate_prf(ret, is_tuple=True)

        def calculate_tuple_metrics_one_label(sent_gold_tuples, sent_pred_tuples, omit_label=True):
            sentence_tuple_metrics = [calculate_sentence_tuple_metrics(g, p, omit_label) for g, p in
                                      zip(sent_gold_tuples, sent_pred_tuples)]

            # calculate TP, FP, FN for all sentences
            ret = {
                k: sum(s_metrics[k] for s_metrics in sentence_tuple_metrics)
                for k in [
                    'E-TP', 'E-FP', 'E-FN',  # exact match
                    'B-TP', 'B-FP', 'B-FN',  # binary match
                ]
            }

            return calculate_prf(ret, is_tuple=True)

        def unique_tuple(list_of_dicts):
            def _str(tup):
                return f'{" ".join(tup["subject"])}_{" ".join(tup["object"])}_{" ".join(tup["aspect"])}_{" ".join(tup["predicate"])}_{tup["label"]}'

            unique_dicts = set()
            unique_dicts_list = []

            for d in list_of_dicts:
                s = _str(d)

                if s not in unique_dicts:
                    unique_dicts.add(s)
                    unique_dicts_list.append(d)

            return unique_dicts_list

        ret = {
            # tuple of four result
            **get_and_add_infix(calculate_tuple_metrics_one_label(sent_gold_tuples, sent_pred_tuples, omit_label=True),
                                'T4'),
        }

        # get score for each label
        label_results = [
            calculate_tuple_metrics_one_label(
                [unique_tuple(t for t in gold_tuples if t['label'] == k) for gold_tuples in sent_gold_tuples],
                [unique_tuple(t for t in pred_tuples if t['label'] == k) for pred_tuples in sent_pred_tuples],
                omit_label=False,
            ) for k in ALL_LABELS
        ]

        # add label results to final result
        for k, r in zip(ALL_LABELS, label_results):
            ret.update(
                get_and_add_infix(r, f'T5-{k}')
            )

        # calculate micro average P, R, F1
        ret.update(
            get_and_add_infix(
                calculate_prf({
                    k: sum(e_metrics[k] for e_metrics in label_results)
                    for k in [
                        'E-TP', 'E-FP', 'E-FN',  # exact match
                        'B-TP', 'B-FP', 'B-FN',  # binary match
                    ]
                }, is_tuple=True),
                'T5-MICRO'
            )
        )

        # calculate macro average P, R, F1
        ret.update(
            get_and_add_infix(
                {
                    k: sum(e_metrics[k] for e_metrics in label_results) / len(label_results)
                    for k in [
                    'E-P', 'E-R', 'E-F1',  # exact match
                    'B-P', 'B-R', 'B-F1',  # binary match
                ]
                },
                'T5-MACRO'
            )
        )

        return ret

    def read_file(f_name):
        data = []
        with open(f_name, 'r', encoding="utf-8") as f:
            sent_tuples = []
            txt = False
            for l in f:
                l = l.strip()

                if len(l) == 0:
                    if txt:
                        data.append(sent_tuples)
                    sent_tuples = []
                    txt = False
                elif l.startswith('{'):
                    sent_tuples.append(json.loads(l))
                else:
                    # text line
                    txt = True

            if txt:
                data.append(sent_tuples)

        return data

    def round_result(d):
        return {k: round(d[k], 4) for k in d}

    test_files = os.listdir(gold_dir)

    # Check if each file in the test_files exists
    for f_name in test_files:
        if not os.path.exists(os.path.join(pred_dir, f_name)):
            raise ValueError(f'The file "{f_name}" does not exist.')

    sent_gold_tuples = []
    sent_pred_tuples = []
    for f_name in test_files:
        gold_tuples = read_file(os.path.join(gold_dir, f_name))
        pred_tuples = read_file(os.path.join(pred_dir, f_name))

        if len(gold_tuples) != len(pred_tuples):
            raise ValueError(f'The file number of sentences in "{f_name}" does not match.')
        else:
            sent_gold_tuples.extend(gold_tuples)
            sent_pred_tuples.extend(pred_tuples)

    return round_result({
        **calculate_ner_metrics(sent_gold_tuples, sent_pred_tuples),
        **calculate_tuple_metrics(sent_gold_tuples, sent_pred_tuples)
    })


if __name__ == "__main__":
    # gold_dir = 'dev' to folder containing golden
    gold_dir = 'folder/gold'
    pred_dir = 'folder/pred'

    print(evaluate_intra(gold_dir, pred_dir))
    df = pd.DataFrame.from_dict(evaluate_intra(gold_dir, pred_dir), orient='index', columns=['Value'])
    # Lưu DataFrame vào một file Excel
    file_path = "evaluation.xlsx"  # Đường dẫn đến tệp Excel
    df.to_excel(file_path)

    print(f"Dữ liệu đã được lưu vào {file_path}.")
