package com.example.dessusdi.myfirstapp.recycler_view;

import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.dessusdi.myfirstapp.MainActivity;
import com.example.dessusdi.myfirstapp.R;
import com.example.dessusdi.myfirstapp.fragments.DetailsFragment;
import com.example.dessusdi.myfirstapp.models.air_quality.WaqiObject;

import java.util.List;


public class AqcinListAdapter extends RecyclerView.Adapter<AqcinCellView> {

    private final List<WaqiObject> list;
    private final Context context;

    /**
     * @param list WaqiObject list
     * @param context context of the activity
     * @see WaqiObject
     */
    public AqcinListAdapter(List<WaqiObject> list, Context context) {
        this.list = list;
        this.context = context;
    }

    /**
     * Setting up Aqcin cell view.
     * @param viewGroup aqcin cell view
     * @param position position of the cell in the recycler view
     * @return cell view
     * @see AqcinCellView
     */
    @Override
    public AqcinCellView onCreateViewHolder(final ViewGroup viewGroup, final int position) {
        View view = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.card_view, viewGroup, false);
        final AqcinCellView aqcinView = new AqcinCellView(view);
        aqcinView.setContext(this.context);

        // On click on the cell view
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final int position = aqcinView.getAdapterPosition();

                DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
                float dpWidth = displayMetrics.widthPixels / displayMetrics.density;

                // Testing if layout support multiple screen
                if(dpWidth < 800.0) {
                    // Doesn't support so replace with a new fragment
                    DetailsFragment fragment = new DetailsFragment();
                    fragment.setCity(list.get(position));
                    FragmentTransaction transaction = ((MainActivity) context).getFragmentManager().beginTransaction();
                    transaction.replace(R.id.main_container, fragment);
                    transaction.setTransition(FragmentTransaction.TRANSIT_FRAGMENT_OPEN);
                    transaction.addToBackStack(null);
                    transaction.commit();
                } else {
                    // Support multiple screen so use current details fragment
                    FragmentManager fragmentManager = ((MainActivity) context).getFragmentManager();
                    DetailsFragment detailsFragment = (DetailsFragment) fragmentManager.findFragmentById(R.id.detail_fragment);
                    detailsFragment.setCity(list.get(position));
                    detailsFragment.fetchCityInformation();
                }
            }
        });

        return aqcinView;
    }

    /**
     * Bind WaqiObject to cell view
     * @param aqcinCellView aqcin cell view
     * @param index position of the cell in the recycler view
     * @see AqcinCellView
     * @see WaqiObject
     */
    @Override
    public void onBindViewHolder(AqcinCellView aqcinCellView, int index) {
        WaqiObject myObject = list.get(index);
        aqcinCellView.setWaqiObject(myObject);
    }

    /**
     * @return List size (number of stations)
     */
    @Override
    public int getItemCount() {
        return list.size();
    }
}
