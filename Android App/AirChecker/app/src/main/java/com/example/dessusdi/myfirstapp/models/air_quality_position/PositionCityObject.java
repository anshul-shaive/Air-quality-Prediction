package com.example.dessusdi.myfirstapp.models.air_quality_position;

import java.util.ArrayList;


public class PositionCityObject {
    private ArrayList<Float> geo;
    private String name;
    private String url;

    public ArrayList<Float> getGeo() {
        return geo;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
