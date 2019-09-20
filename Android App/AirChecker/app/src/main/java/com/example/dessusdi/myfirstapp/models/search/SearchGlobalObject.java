package com.example.dessusdi.myfirstapp.models.search;

import java.util.ArrayList;


public class SearchGlobalObject {
    private String status;
    private ArrayList<SearchLocationObject> data;

    public ArrayList<SearchLocationObject> getData() {
        return data;
    }

    public void setData(ArrayList<SearchLocationObject> data) {
        this.data = data;
    }
}
