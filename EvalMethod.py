def classification_scores(gts, preds, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13]):
    from sklearn import metrics
    import warnings
    warnings.filterwarnings("ignore")
    accuracy        = metrics.accuracy_score(gts,  preds)
    f1_micro        = metrics.f1_score(gts,        preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro    = metrics.recall_score(gts,    preds, average='micro')
    f1_macro        = metrics.f1_score(gts,        preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro    = metrics.recall_score(gts,    preds, average='macro')

    # class wise score
    f1s        = metrics.f1_score(gts,        preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls    = metrics.recall_score(gts,    preds, average=None)

    confusion = metrics.confusion_matrix(gts,preds, labels=labels)

    print("accuracy: ", accuracy)
    print("f1_micro: ", f1_micro)
    print("precision_micro: ", precision_micro)
    print("recall_micro: ", recall_micro)
    print("f1_macro: ", f1_macro)
    print("precision_macro: ", precision_macro)
    print("recall_macro: ", recall_macro)
    print("f1s: ", f1s)
    print("precisions: ", precisions)
    print("recalls: ", recalls)
    print("confusion: ", confusion)

    print("classification_report")
    print(metrics.classification_report(gts, preds, digits=4))