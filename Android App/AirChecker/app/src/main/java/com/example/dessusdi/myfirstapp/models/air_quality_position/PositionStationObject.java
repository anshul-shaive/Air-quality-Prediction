package com.example.dessusdi.myfirstapp.models.air_quality_position;


public class PositionStationObject {
    private int aqi = 0;
    private int idx = 0;
    private String dominentpol = "";
    private PositionCityObject city;

    public int getAqi() {
        return aqi;
    }

    public PositionCityObject getCity() {
        return city;
    }

    public void setCity(PositionCityObject city) {
        this.city = city;
    }
}
