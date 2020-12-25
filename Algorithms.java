package JavaRush;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.TreeMap;

public class Algorithms {
    public static void main(String[] args) {
        
    }

    public static void bubbleSort(int[] array) {
        for(int i=array.length -1; i>1; i--){
            for(int j = 0; j < i; j++ ){
                if (array[j] > array[j+1]){
                    int temp = array[j];
                    array[j] = array[j+1];
                    array[j+1] = temp;
                }
            }
        }
    }
    
    public static void sortBySelect(int[] array){
        for(int i = 0; i < array.length-1; i++){
            int min = i;
            for(int j = i +1; j < array.length; j++){
                if (array[j] < array[min]) {
                    min = j;
                }
            }
            int temp = array[i];
            array[i] = array[min];
            array[min] = temp;
        }
    }
    
    public static void insertionSort(int[] array){
        for (int i = 1; i < array.length; i++){
            int temp = array[i];
            int j = i;
            while(j > 0 && array[j - 1] >= temp){
                array[j] = array[j-1];
                --j;
            }
            array[j] = temp;
        }
    }
    
    public static void shellSort(int[] array){
        int length = array.length;
        int step = length / 2;
        while (step > 0){
            for (int numberOfGroup = 1; numberOfGroup < length - step; numberOfGroup++){
                int j = numberOfGroup;
                    while(j >= 0 && array[j] > array[j+ step]){
                        int temp = array[j];
                        array[j] = array[j + step];
                        array[j + step] = temp;
                        j--;
                    }
            }
            step = step/2;
        }
    }
    
    public static void recursionFastSort(int[] array, int min, int max){
        if (array.length == 0)
            return;
        if (min >= max)
            return;
        
        int middle = min + (max - min)/2;
        
        int middleElement = array[middle];
        
        int i = min, j = max;
        while (i <= j){
            while(array[i] < middleElement){
                i++;
            }
            while(array[j] > middleElement){
                j--;
            }
            if (i <= j){
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
                i++;
                j--;
            }
        }
        if (min < j)
            recursionFastSort(array, min, j);
        if (max > i)
            recursionFastSort(array, i, max);
    }
    
    public static int[] mergeSort(int[] array1){
        int[] sortArr = Arrays.copyOf(array1, array1.length);
        
        int[] bufferArr = new int[array1.length];
        
        return recurtionMergeSort(sortArr, bufferArr, 0, array1.length);
    }
    
    public static int[] recurtionMergeSort(int[] sortArr, int[] bufferArr, int startIndex, int endIndex){
        if (startIndex >= endIndex -1){
            return sortArr;
        }
        int middle = startIndex + (endIndex - startIndex) / 2;
        int[] firstSortArr = recurtionMergeSort(sortArr, bufferArr, startIndex, middle);
        int[] secondSortArr = recurtionMergeSort(sortArr, bufferArr, middle, endIndex);
        
        int firstIndex = startIndex;
        int secondIndex = middle;
        int destIndex = startIndex;
        int[] result = firstSortArr == sortArr ? bufferArr : sortArr;
        while(firstIndex < middle && secondIndex < endIndex) {
            result[destIndex++] = firstSortArr[firstIndex] < secondSortArr[secondIndex]
                    ? firstSortArr[firstIndex++] : secondSortArr[secondIndex++];
        }
        while(firstIndex < middle){
            result[destIndex++] = firstSortArr[firstIndex++];
        }
        while(secondIndex < endIndex){
            result[destIndex++] = secondSortArr[secondIndex++];
        }
        
        return result;
    }
    
    public static void fillBackpack(Bag bag, List<Item> items){
        for (Item item : items){
            if(bag.getMaxWeight() > bag.getCurrentWeight() + item.getWeight()){
                bag.addItem(item);
            }
        }
    }
    
    public static void effectiveFillBackpack(Bag bag, List<Item> items){
        Map<Double, Item> sortByRatio = new TreeMap(Collections.reverseOrder());
        for(Item item : items){
            sortByRatio.put((double)item.getCost() / item.getWeight(), item);
        }
        for (Map.Entry<Double, Item> entry : sortByRatio.entrySet()){
            if(bag.getMaxWeight() > bag.getCurrentWeight() + entry.getValue().getWeight()){
                bag.addItem(entry.getValue());
            }
        }
    }

}

class Item implements Comparable<Item> {
    private String name;
    private int weight;
    private int cost;
    
    public Item(String name, int weight, int cost){
        this.name = name;
        this.weight = weight;
        this.cost = cost;
    }
    
    public String getName(){
        return name;
    }
    public int getWeight(){
        return weight;
    }
    public int getCost() {
        return cost;
    }
    @Override
    public int compareTo(Item o){
        return this.cost > o.cost ? -1 : 1;
    }
}

class Bag {
    private final int maxWeight;
    private List<Item> items;
    private int currentWeight;
    private int currentCost;
    
    public Bag(int maxWeight){
        this.maxWeight = maxWeight;
        items = new ArrayList<>();
        currentCost = 0;
    }
    public int getMaxWeight(){
        return maxWeight;
    }
    public int getCurrentCost(){
        return currentCost;
    }
    public int getCurrentWeight(){
        return currentWeight;
    }
    
    public void addItem(Item item){
        items.add(item);
        currentWeight += item.getWeight();
        currentCost += item.getCost();
    }
}

class Graph {
    private final int MAX_VERTS = 10;
    private Vertex vertexArray[];
    //массив вершин
    private int adjMat[][];
    //матрица смежности
    
    private int nVerts;
    // текущее количество вершин
    
    private Stack stack;
    
    public Graph(){
// инициализация внутрених полей

        vertexArray = new Vertex[MAX_VERTES];
        adjMat = new int[MAX_VERTS][MAX_VERTS];
        nVerts = 0;
        for (int j=0; j<MAX_VERTS; j++){
            for(int k=0; k<MAX_VERTS; k++){
                adjMat[j][k] = 0;
            }
        }
        stack = new Stack();
    }
    
    public void addVertex(char lab){
        vertexArray[nVerts++] = new Vertex(lab);
    }
    public void addEdge(int start, int end){
        adjMat[start][end] = 1;
        adjMat[end][start] = 1;
    }
    public void displayVertex(int v){
        System.out.println(vertexArray[v].getLabel());
    }
    public void dfs() {
// обход в глубину

        vertexArray[0].setWasVisited(true);
// берется первая вершина

        displayVertex(0);
        stack.push(0);
        
        while (!stack.empty()){
            int v = getAdjUnvisitedVertex(stack.peek());
// вынуть индекс смежной вершины, если есть 1, нету -1
            
            if(v == -1){
// если непройденных смежных вершин нету

                stack.pop();
// элемент извлекается из стека
            } else {
                vertexArray[v].setWasVisited(true);
                displayVertex(v);
                stack.push(v);
// элемент попадает на вершину стека
            }
        }
        for(int j=0; j<nVerts; j++){
// сброс флагов
            vertexArray[j].wasVisited = false;
        }
    }
    private int getAdjUnvisitedVertex(int v){
        for(int j=0; j<nVerts; j++){
            if (adjMat[v][j] == 1 && vertexArray[j].wasVisited == false){
                return j;
//возвращает первую найденную вершину
            }
        }
        return -1;
    }
    
}
