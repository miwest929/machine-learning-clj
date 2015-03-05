(ns naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [naive-bayes.core :refer :all]))

(def dataset [[600] [470] [170] [430] [300]])

(deftest mean-test
  (testing "definition of mean function"
    (is (= (mean dataset 0) 394))))

(deftest variance-test
  (testing "definition of the variance function"
    (is (= (variance dataset 0 (mean dataset 0)) 21704.0))))

; 9 + 1 + 0 + 25 + 1 
