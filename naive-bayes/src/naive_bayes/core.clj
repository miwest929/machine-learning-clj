(ns naive-bayes.core
  (:require [clojure-csv.core :as csv] [clojure.java.io :as io] [clojure.set :as clj-st]))

(defn argmax
  "Returns the argument in a collection that produces the
   maximum result of a function f"
  [f coll]
  (reduce (fn [a b] (if (> (f a) (f b)) a b)) coll))
(defn interchange-rows
  "Produce a new matrix 'M' with rows 'a' and 'b' swapped"
  [M a b]
  (assoc M b (M a) a (M b)))

(def people-csv "../data/people.csv")
(defn load-dataset
  "Takes csv filename of the dataset as input and returns
   a List of the training examples"
  [fname]
  (with-open [file (io/reader fname)]
   (csv/parse-csv (slurp file))))
(defn prepare-example [[gender age-value height-value weight-value]] [gender (Float/parseFloat age-value) (Float/parseFloat height-value) (Float/parseFloat weight-value)])
(def examples (map prepare-example (load-dataset people-csv)))
(def training-set (take (/ (count examples) 2) (shuffle examples))) ; Half for training
(def test-set (vec (clj-st/difference (set examples) (set training-set))))

(def gender 0)
(def age 1)
(def height 2)
(def weight 3)

(defn get-feature [example feature] (nth example feature))

(defn mean [examples feature] (int (/ (reduce + (map #(get-feature % feature) examples)) (count examples))))

(defn variance [examples feature mean-value]
  (/ (reduce + (map #(Math/pow (- (get-feature % feature) mean-value) 2) examples)) (count examples)))

(defn distribution [examples feature] 
  (let [mean-value (mean examples feature)]
    [mean-value (variance examples feature mean-value)]))

(defn probability [subset examples] (double (/ (count subset) (count examples))))
(defn probability-feature [[mean-val variance-val] given-val]
   (* (/ 1 (Math/sqrt (* 2 Math/PI variance-val)))
      (Math/exp (/ (- (Math/pow (- given-val mean-val) 2)) (* 2 variance-val)))))

; (classify (naive-bayes-classifier examples gender) test-ex)
(defn naive-bayes-classifier
  "Given a set of training examples and the category column
   generate a Naive Bayes Classifier"
  [training category-idx]
  (let [males (filter #(= "m" (nth % category-idx)) training)
        females (filter #(= "f" (nth % category-idx)) training)]
    {:prob-male-given (fn [test-ex]
       (* (probability males training)
          (probability-feature (distribution males age) (get-feature test-ex age))
          (probability-feature (distribution males height) (get-feature test-ex height))
          (probability-feature (distribution males weight) (get-feature test-ex weight))))
     :prob-female-given (fn [test-ex]
       (* (probability females training)
          (probability-feature (distribution females age) (get-feature test-ex age))
          (probability-feature (distribution females height) (get-feature test-ex height))
          (probability-feature (distribution females weight) (get-feature test-ex weight))))
     }))
(defn classify
  "Given a prepared Naive Bayes Classifier and a test set
   returns the classified category"
  [classifier test-ex]
  (let [prob-male ((classifier :prob-male-given) test-ex)
        prob-female ((classifier :prob-female-given) test-ex)]
    (if (> prob-male prob-female) "m" "f")))
(defn -main
  "I don't do a whole lot."
  []
  (let [classifier (naive-bayes-classifier training-set gender)
        results (map #(identity {:example % :expected (first %) :actual (classify classifier %)}) test-set)
        wrong (filter #(not= (% :expected) (% :actual)) results)]
    (println "Out of " (count results) " the classifier classified " (count wrong) " incorrectly. Accuracy is " (/ (count wrong) (count results)))))
