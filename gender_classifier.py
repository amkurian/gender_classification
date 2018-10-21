from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43], [168, 75, 41], [168, 77, 41]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female', 'female']

test_data = [[168, 77, 40],[154, 65, 37],[181,60,38]]

dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(X,Y)
prediction = dtc_clf.predict(test_data)
print prediction


rfc_clf = RandomForestClassifier()
rfc_clf.fit(X,Y)
rfc_prediction = rfc_clf.predict(test_data)
print rfc_prediction